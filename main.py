import argparse
from pathlib import Path

from gensim.models import Doc2Vec
from sklearn import cluster

import resources.constant
from util.filesystem_validators import AccessibleDirectory, AccessibleTextFile


def main():
    parser = _initialize_parser()
    args = parser.parse_args()

    model = Doc2Vec.load(args.input)

    x = model.docvecs.vectors_docs

    print("[Cluster-embedded-entities] Starting clustering")

    kmeans = cluster.KMeans(n_clusters=resources.constant.NUM_CLUSTERS,
                            algorithm="elkan",
                            init="k-means++",
                            n_jobs=resources.constant.NUMBER_OF_PARALLEL_EXECUTIONS)
    kmeans.fit(x)

    print("[Cluster-embedded-entities] Clustering finished")
    print("[Cluster-embedded-entities] Start sorting entities based on assigned cluster")

    cluster_map = {k: [] for k in range(resources.constant.NUM_CLUSTERS)}

    for i, word in enumerate(model.docvecs.doctags):
        cluster_map[kmeans.labels_[i]].append(word)

    print("[Cluster-embedded-entities] Sorting finished")
    print("[Cluster-embedded-entities] Start writing to file")

    with open(Path(args.output, resources.constant.OUTPUT_FILE), "w+") as output_file:
        for i in range(resources.constant.NUM_CLUSTERS):
            closest_word = model.wv.similar_by_vector(kmeans.cluster_centers_[i], 1)[0][0]
            closest_entity = model.docvecs.most_similar([kmeans.cluster_centers_[i], 1])[0][0]
            print(f"[[CLUSTER {i}]] - Closest word: {closest_word}; Closest entity: {closest_entity}", end="\n",
                  file=output_file)
            print(*(cluster_map[i]), sep="\n", file=output_file, end="\n")


def _initialize_parser():
    general_parser = argparse.ArgumentParser(description='Clustering trained entity embeddings')
    general_parser.add_argument("--input", help='gensim model containing embedded entities',
                                action=AccessibleTextFile, required=True)
    general_parser.add_argument("--output", help='Desired location for storing cluster information', required=True,
                                action=AccessibleDirectory)

    return general_parser


if __name__ == "__main__":
    main()
