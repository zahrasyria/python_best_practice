CHUNK_SIZE_EACH = 44000



def __embedding(text):
  return compute_mean(embedding_fn(model, text))



def compute_bert_embeddings(dataframe_chunk, current_index, end_marker):

  np_chunk = __embedding(dataframe_chunk.loc[current_index * end_marker]['Title']).detach().numpy()
  #np_chunk = np_chunk.reshape(np_chunk.shape[1])

  for idx in range(1, end_marker):

    try:
      embedding = __embedding(dataframe_chunk.loc[(current_index * end_marker) + idx]['Title']).detach().numpy()
      #embedding = embedding.reshape(embedding.shape[1])
      np_chunk = np.append(np_chunk, embedding, axis = 0)
      print('\r {}'.format(np_chunk.shape), end = '')
    except Exception as e:
      print(e)
      np_chunk = np.append(np_chunk, np.zeros(shape = (1, 768)), axis = 0)
      continue 

  print(np_chunk.shape)
  np.savez_compressed('title_{}'.format(current_index), a = np_chunk)


def compute_embeddings_and_save(dataframe):

  n_rows = len(dataframe)
  
  chunk_sizes = n_rows // CHUNK_SIZE_EACH
  remaining = n_rows - chunk_sizes * CHUNK_SIZE_EACH

  for i in range(1):

    compute_bert_embeddings(dataframe[i * CHUNK_SIZE_EACH : (i * CHUNK_SIZE_EACH) + CHUNK_SIZE_EACH ], i, CHUNK_SIZE_EACH)

