import argparse
from pathlib import Path

import nltk

import resources.constant
from util.filesystem_validators import AccessibleDirectory, AccessibleTextFile
from gensim.models import Doc2Vec
from nltk.cluster import KMeansClusterer


def main():

    parser = _initialize_parser()
    args = parser.parse_args()

    model = Doc2Vec.load(args.input)

    x = model[model.docvecs.doctags]

    print("[Cluster-embedded-entities] Starting clustering")

    clusterer = KMeansClusterer(resources.constant.NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
    assigned_clusters = clusterer.cluster(x, assign_clusters=True)

    print("[Cluster-embedded-entities] Clustering finished")
    print("[Cluster-embedded-entities] Start sorting entities based on assigned cluster")

    cluster_map = dict.fromkeys(range(resources.constant.NUM_CLUSTERS), [])

    for i, word in enumerate(model.docvecs.doctags):
        cluster_map[assigned_clusters[i]].append(word)

    print("[Cluster-embedded-entities] Sorting finished")
    print("[Cluster-embedded-entities] Start writing to file")

    with open(Path(args.output, resources.constant.OUTPUT_FILE), "w+") as output_file:
        for i in range(resources.constant.NUM_CLUSTERS):
            print(f"[[CLUSTER {i}]]", end="\n", file=output_file)
            print(*cluster_map[i], sep="\n", file=output_file, end="\n")


def _initialize_parser():
    general_parser = argparse.ArgumentParser(description='Clustering trained entity embeddings')
    general_parser.add_argument("--input", help='gensim model containing embedded entities',
                                action=AccessibleTextFile, required=True)
    general_parser.add_argument("--output", help='Desired location for storing cluster information', required=True,
                                action=AccessibleDirectory)

    return general_parser


if __name__ == "__main__":
    main()
