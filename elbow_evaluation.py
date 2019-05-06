import os

for x in range(100, 100000, 100):
    print(f"Start clustering for k={x}")
    os.system(f"python3 main.py --input=/san2/data/teaching/chiki/wiki_living_people_model/doc2vec.binary.model --output=/san2/data/teaching/chiki/clustering_results --k={x}")

