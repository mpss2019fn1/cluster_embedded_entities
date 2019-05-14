import argparse
import logging

from clusterers.dbscan_clusterer import DBScanClusterer
from clusterers.kmeans_clusterer import KMeansClusterer
from util.filesystem_validators import AccessibleDirectory, AccessibleTextFile
from writers.csv_cluster_writer import CSVClusterWriter
from writers.text_cluster_writer import TextClusterWriter

VALID_OUTPUT_MODES = ["csv", "text"]


def main():
    logging.basicConfig(format='%(asctime)s : [%(threadName)s] %(levelname)s : %(message)s', level=logging.INFO)
    parser = _initialize_parser()
    args = parser.parse_args()

    if 'action' not in args or not args.action:
        parser.print_usage()
        return

    if args.action == 'kmeans':
        _clustering_kmeans(args)
    if args.action == 'dbscan':
        _clustering_dbscan(args)


def _clustering_kmeans(args):
    clusterer = KMeansClusterer(args.input, args.threads, args.k)
    clusterer.build_clusters()
    _create_writer(args).write(clusterer, args.output if "output" in args else None)


def _clustering_dbscan(args):
    clusterer = DBScanClusterer(args.input, args.threads, args.eps)
    clusterer.build_clusters()
    _create_writer(args).write(clusterer, args.output if "output" in args else None)


def _create_writer(args):
    if args.output_mode == "text":
        return TextClusterWriter()
    if args.output_mode == "csv":
        return CSVClusterWriter()
    raise Exception(f"Invalid output type arguments supplied. Chose from {', '.join(VALID_OUTPUT_MODES)}")


def _initialize_parser():
    general_parser = argparse.ArgumentParser(description='Clustering trained entity embeddings')
    subparsers = general_parser.add_subparsers()

    _initialize_kmeans_parser(subparsers)
    _initialize_dbscan_parser(subparsers)

    return general_parser


def _initialize_kmeans_parser(subparsers):
    kmeans_parser = subparsers.add_parser('kmeans', help='Use k-means for clustering')
    kmeans_parser.set_defaults(action='kmeans')
    kmeans_parser.add_argument("--input", help='gensim model containing embedded entities',
                               action=AccessibleTextFile, required=True)
    kmeans_parser.add_argument("--k", help='number of clusters to build', required=True, type=int)
    kmeans_parser.add_argument("--output", help='Desired location for storing cluster information',
                               action=AccessibleDirectory)
    kmeans_parser.add_argument("--output-mode",
                               help=f"Define the type of output. Chose from: {', '.join(VALID_OUTPUT_MODES)}",
                               required=True)
    kmeans_parser.add_argument("--threads", help="Number of threads to use", default=8, type=int)


def _initialize_dbscan_parser(subparsers):
    dbscan_parser = subparsers.add_parser('dbscan', help='Use DBSCAN for clustering')
    dbscan_parser.set_defaults(action='dbscan')
    dbscan_parser.add_argument("--input", help='gensim model containing embedded entities',
                               action=AccessibleTextFile, required=True)
    dbscan_parser.add_argument("--eps", help='eps for expanding clusters', required=True, type=float)
    dbscan_parser.add_argument("--output", help='Desired location for storing cluster information',
                               action=AccessibleDirectory)
    dbscan_parser.add_argument("--output-mode",
                               help=f"Define the type of output. Chose from: {', '.join(VALID_OUTPUT_MODES)}",
                               required=True)
    dbscan_parser.add_argument("--threads", help="Number of threads to use", default=8, type=int)


if __name__ == "__main__":
    main()
