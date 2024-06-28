import os
import sys
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from itertools import chain
import torch
import torch.distributed
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, GenerationConfig
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
import time

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

def normalize_speaker(speaker_in):
  """Add '_' before and after speaker name if it does not contain it already"""
  if speaker_in == '-' or speaker_in == '__':
    return '_'

  speaker = speaker_in.replace(' ', '_')
  speaker = speaker.strip()

  if not speaker.startswith('_'):
    speaker = '_'+speaker
  if not speaker.endswith('_'):
    speaker = speaker+'_'
  return speaker


def match_mention_state(m, inputs, maps, position=None, debug=False, start_index=0):
  if '##' in m:
    index_num = m.index('##')
  else:
    if not m[0].startswith('['):
      print('get_chains::error ## not in split', m)
    index_num = len(m)

  if ']]' in inputs:
    end_index = inputs.index(']]')
  elif '**' in inputs:
    end_index = inputs.index('**')
  else:
    end_index = len(inputs)

  # m_clean = [x for x in m if x != '##']
  m_clean = []
  for x in m:
    if x != '##':
      m_clean.append(x)
    if x == '**':
      break

  # get context
  context = []
  found_num = False
  for s in m:
    if found_num:
      context.append(s)
    if '##' == s:
      found_num = True

  maps_index = 0
  indices = []
  for i in range(start_index, end_index):
    maps_index = i
    if inputs[i] == m_clean[0]:
      if inputs[i:i+len(m_clean)] == m_clean:
        indices.append((maps[maps_index], maps[maps_index + index_num  - 1]))

        if maps[maps_index + index_num  - 1] == -1:
          print('index negative', maps[maps_index:], ' index_num',  index_num)
          print('index negative', inputs[i:], ' index_num',  index_num)
          print(f'i {i} maps_index {maps_index}')


  if len(indices) == 0:
    print('none found match_mention', m)
    print('inputs', inputs)
    return []
  elif len(indices) > 1 and debug:
    print('match_mention: too many ', m,  indices, 'm_clean - use both')

  if (-1,-1) in indices:
    print('error for ',m, indices)
    return []

  return indices


def match_link_state(link, inputs, maps, cluster_name_to_cluster,
                     debug=True, node_wise=True):
  link_mentions = [m.split(' ') for m in link]
  links = []
  if len(link_mentions) == 1 and node_wise:
    m0 = link_mentions[0]
    try:
      index_m0 = match_mention_state(m0, inputs, maps, position=None)
      links = [index_m0]
    except Exception as e:
      print(str(e))
    return links


  m0 = link_mentions[0]
  m1 = link_mentions[1]

  if debug:
    print('match_link', m0, m1)

  # invert indices
  if m1[0].startswith('[') and len(m1[0]) > 0:
    cluster = cluster_name_to_cluster.get(m1[0], None)
    if cluster is not None:
      index_m1 = [cluster[-1]]
    else:
      print('cluster does not exists')
      return []
  else:
    index_m1 = match_mention_state(m1, inputs, maps, position=None)


  if debug:
    print(index_m1 ,'match' ,m1)

  if len(index_m1) > 1:
    print('index_m1', index_m1)

  try:
    index_m0 = match_mention_state(m0, inputs, maps, position=None)
  except Exception as e:
    print('error', str(e))
    index_m0 = []

  if debug:
    print(index_m0 ,'match' , m0)

  if len(index_m0) > 1:
    print('index_m0', index_m0)

  if len(index_m1) > 0 and len(index_m0) > 0:
      i1 = index_m1[-1]
      i2 = index_m0[-1]
      links.append([i1, i2])

  # use only last link
  if len(links) > 1:
    print('too many links, ', links, 'for link', link)
    print('context', inputs)

    return links[-1:]

  return links


def get_mentions_for_link_state(link, node_wise):
  link_split = link.split('->')

  if node_wise and len(link_split) == 1:
    m0 = link_split[0].strip()
    # print('link has only one mention?', link, m0)
    return [m0]

  elif len(link_split) < 2:
    print('link has only one mention - skipping mention', link)
    return []

  if len(link_split) > 2:
    print('link has too many mentions - using first two.', link)
  m0 = link_split[0].strip()
  m1 = link_split[1].strip()
  return [m0, m1]


def get_all_text(states_dict):
  doc_text = {}
  for doc_name, s in states_dict.items():
    pred_clusters = [cluster for _, cluster in s.cluster_name_to_cluster.items()]
    print('predicted clusters with word indexes', pred_clusters)

    text, text_map = [], []
    for k, snt in states_dict[doc_name].input_document['sentences'].items():
      m = states_dict[doc_name].input_document['token_maps'][k]
      text += snt
      text_map += m

    cluster_annotations_start = []
    cluster_annotations_end = []

    # Cluster annotation per token
    for tid in text_map:
      cluster_annotations_start.append([])
      cluster_annotations_end.append([])
      for ci in pred_clusters:
        for m in ci:

          if tid == m[0]:
            m_len = m[1] - m[0]
            name = s.mention_index_to_cluster_name[str(m)]
            cluster_annotations_start[-1].append((name, m_len))

          if tid == m[1]:
            cluster_annotations_end[-1].append(']')

    # get the text with the coreference annotations
    all_text = []
    for tok, start, end in zip(text, cluster_annotations_start, cluster_annotations_end):

      if start:
        for x in sorted(start, key=lambda x : x[1], reverse=True):
          all_text.append('['+str(x[0]))

      all_text.append(tok)

      if end:
        all_text.append(''.join(end))

    doc_text[doc_name] = all_text
  return doc_text


class State(object):
  """Document state."""

  def __init__(self, input_document, model_tokenizer, output_dir, doc_title, node_wise=True, max_len_doc=2048):
    """ Create State object to process documents.

    Args:
      input_document: dictonary with the input document.
      node_wise: Predict mentions too.
      max_len_doc: max sentence pieace tokens, eg. 2000 or 3000 (bit better).

    """
    self.sentence_num = -1
    self.clusters_num = 0

    self.token_map_context, self.annotation_context = [], []
    self.annotation_coreference_start, self.annotation_coreference_end = [], []
    self.token_map, self.annotation = [], []

    # a mention index to cluster mapping, e.g. (23, 24) -> [(23, 24), (41, 42)]
    self.mention_index_to_cluster = {}

    # the first link names the cluster, e.g. (23, 24) -> '1'
    self.mention_index_to_cluster_name = {}
    self.cluster_name_to_cluster = {}

    self.input_document = input_document
    # print('sentence_num', self.sentence_num)
    self.genre = input_document['genres'][0][0]
    self.speakers = {t: spk for (t, spk) in self.input_document['speakers']}

    self.done = False
    self.predictions_str = {}  # keep the predictions
    self.node_wise = node_wise

    self.max_len_doc = max_len_doc
    self.model_tokenizer = model_tokenizer
    self.output_dir = output_dir
    self.doc_title = doc_title

    # move to initial position.
    self.extend()


  def extend_done(self):
    return self.done

  def extend(self, prediction_str=None, use_gold_cluster=False, move=True):

    # move annotation to context
    self.token_map_context +=  self.token_map
    self.annotation_context += self.annotation

    for k in range(len(self.annotation)):
      self.annotation_coreference_start.append([])
      self.annotation_coreference_end.append([])

    assert len(self.annotation_context)  == len(self.annotation_coreference_start)

    self.annotation, self.token_map = [], []

    link_found = False
    if prediction_str is not None and not 'None [' in prediction_str:
      links = [l for l in prediction_str.split(';;') if l != '' ]

      annotation_update = []
      for link in links:
        link_found = True
        link_mentions = get_mentions_for_link_state(link, self.node_wise)

        if len(link_mentions) < 2 and not (self.node_wise and len(link_mentions)):
          print('less mentions as needed skip', link_mentions)
          continue
        indices = match_link_state(link_mentions, self.annotation_full,
                                   self.annotation_full_map,
                                   self.cluster_name_to_cluster,
                                   debug=False)

        if not indices:
          print('not found !!')
          print('indices not found', link, indices)
          print('self.annotation_full', self.annotation_full )
          print('annotation + context', self.get_input_annotation())
          continue

        if True:
          index = indices[0]
          cluster = []
          for mention_index in index:
            if str(mention_index) in self.mention_index_to_cluster:
              cluster = self.mention_index_to_cluster[str(mention_index)]
              break


          if not cluster:
            self.clusters_num += 1
            cluster_name = str(self.clusters_num)

            if use_gold_cluster:  # just to evaluate on gold

              for ni, cx in enumerate(self.input_document['clusters']):
                for mx in cx:
                  if mx in index:
                    cluster_name = str(ni+1)
                    break

          else:
            cluster_name = self.mention_index_to_cluster_name[str(cluster[0])]

          for mention_index in index:
            if mention_index not in cluster:
              cluster.append(mention_index)
              self.mention_index_to_cluster[str(mention_index)] = cluster
              self.cluster_name_to_cluster['['+cluster_name] = cluster
              self.mention_index_to_cluster_name[str(mention_index)] = cluster_name
              annotation_update.append([mention_index, cluster_name])

      # update the annotation
      if True:
        for update in annotation_update:
          update_index = update[0]
          update_name = update[1]

          for t, coref_starts, coref_end, tid in zip(self.annotation_context,
                                    self.annotation_coreference_start,
                                    self.annotation_coreference_end,
                                    self.token_map_context):



            if update_index[0] == tid:
              coref_starts.append(update)
              coref_starts.sort( key=lambda x: x[0][1], reverse=True)


            if update_index[1] == tid:
              coref_end.append(']')

    if move or 'None [' in prediction_str or not link_found:
      self.sentence_num += 1

      if self.sentence_num not in self.input_document['sentences']:
        self.done = True
        return True

      tokens = self.input_document['sentences'][self.sentence_num]
      token_map = self.input_document['token_maps'][self.sentence_num]
      first = True

      for tid, t in zip(token_map, tokens):
        if first:
          self.token_map.append(-1)
          speaker = normalize_speaker(self.speakers[tid])
          self.annotation.append(speaker)
          first = False
        self.token_map.append(tid)
        self.annotation.append(t)

    if self.sentence_num not in self.predictions_str:
      self.predictions_str[self.sentence_num] = ''

    if prediction_str is not None:
      self.predictions_str[self.sentence_num] += prediction_str

    return False

  def input_annotation(self):

    self.annotation_full = ['coref:', self.genre]
    self.annotation_full_map = [-1, -1]
    for t, coref_starts, coref_end, tid in zip(self.annotation_context,
                                  self.annotation_coreference_start,
                                  self.annotation_coreference_end,
                                  self.token_map_context):

      for coref_start in coref_starts:
        coref_name = coref_start[-1]

        self.annotation_full.append('[' + coref_name)
        self.annotation_full_map.append(-1)

      self.annotation_full.append(t)
      self.annotation_full_map.append(tid)

      for end in coref_end:
        coref_name = end[-1]
        self.annotation_full.append(coref_name)
        self.annotation_full_map.append(-1)

    self.annotation_full += ['|'] + self.annotation
    self.annotation_full_map += [-1] + self.token_map
    self.annotation_full += ['**']
    self.annotation_full_map += [-1]


  def encode(self, annotation_str):
    return self.model_tokenizer.encode(annotation_str)

  def get_input_annotation(self, context_right=True):

    self.input_annotation()
    annotation_str = ' '.join(self.annotation_full)

    enc = self.encode(annotation_str)
    shorten = len(enc) > self.max_len_doc

    while len(enc) > self.max_len_doc:   # inefficient ...
      self.annotation_context = self.annotation_context[1:]
      self.token_map_context = self.token_map_context[1:]
      self.annotation_coreference_start = self.annotation_coreference_start[1:]
      self.annotation_coreference_end = self.annotation_coreference_end[1:]

      self.input_annotation()
      annotation_str = ' '.join(self.annotation_full)
      enc = self.encode(annotation_str)

    last_token_id = self.annotation_full_map[-2]  # the last one is **
    self.annotation_context_right = []

    if not shorten and context_right:
      sentence_num = self.sentence_num
      total_len = len(enc)

      while True:
        sentence_num += 1
        if sentence_num not in self.input_document['sentences']:
          break

        first = True
        annotation_context_next = []

        for t, tid in zip(self.input_document['sentences'][sentence_num], self.input_document['token_maps'][sentence_num]):
          if first:
            speaker = normalize_speaker(self.speakers[tid])
            annotation_context_next.append(speaker)
            first = False
          annotation_context_next.append(t)

        annotation_context_right = self.annotation_context_right + annotation_context_next
        enc = self.encode(' '.join(annotation_context_right))

        if (len(enc) + total_len) > self.max_len_doc:
          break
        self.annotation_context_right = annotation_context_right
      if self.annotation_context_right:
        annotation_str = annotation_str + ' ' + ' '.join(self.annotation_context_right)

    enc = self.encode(annotation_str)
    if len(enc) > self.max_len_doc:
      print('warning: document too long', len(enc))

    return annotation_str


def create_document(document: str, tokenizer, title: str = 'not_named'):
  """Creates a datastructure with a title and uses nltk for tokenization.

  Args:
    document: sentences separated with newline ('\n').
    title: the name of the document.

  Returns:
    dict with sentences, maps to token-ids, speakers, and genres.
  """
  input_document = {
      'doc_key': title,
      'sentences': {},
      'token_maps': {},
      'speakers': [],
      'genres': []
  }

  tid = 0
  for k, sentence in enumerate(document.split('\n')):
    # input_document['sentences'][k] = tokenize(text=sentence)
    input_document['sentences'][k] = tokenizer(sentence)
    input_document['token_maps'][k] = []

    for _ in input_document['sentences'][k]:
      input_document['token_maps'][k].append(tid)
      input_document['speakers'].append((tid, '_'))
      input_document['genres'].append(['dn'])
      tid += 1

  return input_document

def create_next_batch(states_dict, batch_size=1, num_batches=1):
  batches = [[]]
  states = []
  for key, state in states_dict.items():
    if state.extend_done():
      continue

    states.append(state)
    if len(states) >= (batch_size * num_batches):
      break
  for state in states:
    batches[-1].append(state.get_input_annotation())
    if len(batches[-1]) >= batch_size:
      if len(batches) >= num_batches:
        break
      batches.append([])
  return states, batches


def predictor_fn(batches, ds_engine, model_tokenizer, generation_config, return_output=True):
    # Tokenize the input text
    inputs = model_tokenizer(batches[0], return_tensors="pt", padding="max_length", truncation=True, max_length=2048).to(device=local_rank)

    # Generate output text
    with torch.no_grad():
        outputs = ds_engine.module.generate(**inputs, generation_config=generation_config, synced_gpus=True)

    if return_output:
      # Decode the generated output text
      output_texts = model_tokenizer.batch_decode(outputs, skip_special_tokens=True, max_length=generation_config.max_length)
      return output_texts

def create_coref(states_dict, ds_engine, world_size, local_rank, batch_size=1, debug_batch_result=False):
  for step in range(10000):  # while states
    states, batches = create_next_batch(states_dict, batch_size, num_batches=1)

    all_done = True
    for rank in range(world_size):
      if rank == local_rank and not states:
        done = torch.tensor(True, dtype=torch.bool).to(device=local_rank)
      else:
        done = torch.tensor(False, dtype=torch.bool).to(device=local_rank)
      deepspeed.comm.broadcast(done, rank)
      if done == False:
        all_done = False
        break

    if all_done:
      break

    if not states:
      print("No documents left to process. Using fill input to synch GPUs.")
      predictor_fn([["Fill"]], ds_engine, model_tokenizer, generation_config, False)
      continue
    else:
      documents_processing = set([x.input_document['doc_key'] for x in states])

      print(f'Processing documents: {documents_processing}')

      results = predictor_fn(batches, ds_engine, model_tokenizer, generation_config)

      for state, result, batch in zip(states, results, batches[0]):
        state.extend(result)

        if debug_batch_result:
          print('input batch: ', batch)
          print('mt5 output: ', result)

if __name__ == "__main__":
  batch_size = 6
  random.seed(42)
  model_name = "mt5-coref-pytorch/link-append-xxl"
  start_time = time.time()
  deepspeed.init_distributed(dist_backend="nccl")
  model_tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
  model_config = T5Config.from_pretrained(model_name)
  model_hidden_size = model_config.d_model

  ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "none"
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 200,
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 1
  }

  dschf = HfDeepSpeedConfig(ds_config)
  if len(sys.argv) < 4:
    model = T5ForConditionalGeneration.from_pretrained(model_name, config=model_config)
  else:
    state_dict = torch.load(sys.argv[3])
    model = T5ForConditionalGeneration.from_pretrained(None, config=model_config, state_dict=state_dict)

  ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
  ds_engine.module.eval()

  generation_config = GenerationConfig.from_pretrained(model_name)
  generation_config.max_length = 384

  tokenizer = lambda x: list(chain.from_iterable([TreebankWordTokenizer().tokenize(word) for word in TreebankWordTokenizer().tokenize(x)]))
  detokenizer = lambda x: TreebankWordDetokenizer().detokenize(x).replace('"', '``')

  input_dir = os.path.abspath(sys.argv[1]).rstrip("/\\")
  output_dir = os.path.abspath(sys.argv[2]).rstrip("/\\")
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

  states_dict = {}

  batch_size_per_gpu = batch_size // world_size
  filelist = os.listdir(input_dir)
  num_files_per_gpu = len(filelist) // world_size
  start_idx = num_files_per_gpu * local_rank
  end_idx = start_idx + num_files_per_gpu if local_rank < world_size - 1 else len(filelist)

  largest_file = 0

  for i, filename in enumerate(filelist):
    f_path = os.path.join(input_dir, filename)
    if os.path.isfile(f_path) and i >= start_idx and i < end_idx:
      print("Processing file: ", f_path)
      f_content = open(f_path, "r", encoding="utf-8").read()
      doc_title = f_path.split("/")[-1].removesuffix(".txt")
      doc_content = create_document(f_content, tokenizer, doc_title)
      states_dict[doc_title] = State(doc_content, model_tokenizer, output_dir, doc_title)

  create_coref(states_dict, ds_engine, world_size, local_rank, batch_size_per_gpu, True)
  doc_text = get_all_text(states_dict)
  for doc_name, text in doc_text.items():
    with open("{}/{}.txt".format(output_dir, doc_name), "w", encoding="utf-8") as output_f:
      output_f.write(str(text))

  print("Time elapsed: ", time.time() - start_time)
