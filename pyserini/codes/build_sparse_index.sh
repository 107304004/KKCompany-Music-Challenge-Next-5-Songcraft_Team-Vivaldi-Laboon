python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/session_song_collection \
  --index indexes/session_song_collection \
  --stemmer None \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw
