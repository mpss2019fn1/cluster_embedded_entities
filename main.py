import argparse
import logging
from pathlib import Path
from typing import Optional

from clustering.DBScan.dbscan_cluster_builder import DBScanClusterBuilder
from clustering.KMeans.kmeans_cluster_builder import KMeansClusterBuilder
from clustering.SimDim.simdim_cluster_builder import SimDimClusterBuilder
from clustering.abstract_cluster_builder import AbstractClusterBuilder
from util.filesystem_validators import WriteableDirectory, ReadableFile
from writers.csv_cluster_writer import CSVClusterWriter
from writers.text_cluster_writer import TextClusterWriter

VALID_OUTPUT_MODES = ["csv", "text"]


def main():
    logging.basicConfig(format="%(asctime)s : [%(threadName)s] %(levelname)s : %(message)s", level=logging.INFO)
    parser = _initialize_parser()
    args = parser.parse_args()

    if "action" not in args or not args.action:
        parser.print_usage()
        return

    cluster_builder: Optional[AbstractClusterBuilder] = None

    if args.action == "kmeans":
        cluster_builder = KMeansClusterBuilder(args.input, args.threads, args.k)

    if args.action == "dbscan":
        cluster_builder = DBScanClusterBuilder(args.input, args.threads, args.eps)

    if args.action == "simdim":
        cluster_builder = SimDimClusterBuilder(args.input, args.threads)

    if not cluster_builder:
        exit(1)

    cluster_builder.build_clusters()
    _create_writer(args).write(cluster_builder, args.output if "output" in args else None)


def _create_writer(args):
    if args.output_mode == "text":
        return TextClusterWriter()
    if args.output_mode == "csv":
        return CSVClusterWriter()

    raise Exception(f"Invalid output type arguments supplied. Choose from {', '.join(VALID_OUTPUT_MODES)}")


def _initialize_parser():
    general_parser = argparse.ArgumentParser(description="Clustering trained entity embeddings")
    subparsers = general_parser.add_subparsers()

    _initialize_kmeans_parser(subparsers)
    _initialize_dbscan_parser(subparsers)
    _initialize_simdim_parser(subparsers)

    return general_parser


def _initialize_kmeans_parser(subparsers):
    kmeans_parser = subparsers.add_parser("kmeans",
                                          help="Use k-means for clustering")
    kmeans_parser.set_defaults(action="kmeans")

    kmeans_parser.add_argument("--input",
                               help="gensim model containing embedded entities",
                               type=Path,
                               action=ReadableFile,
                               required=True)
    kmeans_parser.add_argument("--k",
                               help="number of clusters to build",
                               required=True,
                               type=int)
    kmeans_parser.add_argument("--output",
                               help="Desired location for storing cluster information",
                               type=Path,
                               action=WriteableDirectory,
                               required=True)
    kmeans_parser.add_argument("--output-mode",
                               help=f"Define the type of output. Choose from: {', '.join(VALID_OUTPUT_MODES)}",
                               required=True)
    kmeans_parser.add_argument("--threads",
                               help="Number of threads to use",
                               type=int,
                               default=8)


def _initialize_dbscan_parser(subparsers):
    dbscan_parser = subparsers.add_parser("dbscan",
                                          help="Use DBSCAN for clustering")
    dbscan_parser.set_defaults(action="dbscan")

    dbscan_parser.add_argument("--input",
                               help="gensim model containing embedded entities",
                               type=Path,
                               action=ReadableFile,
                               required=True)
    dbscan_parser.add_argument("--eps",
                               help="eps for expanding clusters",
                               type=float,
                               required=True)
    dbscan_parser.add_argument("--output",
                               help="Desired location for storing cluster information",
                               type=Path,
                               action=WriteableDirectory,
                               required=True)
    dbscan_parser.add_argument("--output-mode",
                               help=f"Define the type of output. Choose from: {', '.join(VALID_OUTPUT_MODES)}",
                               required=True)
    dbscan_parser.add_argument("--threads",
                               help="Number of threads to use",
                               type=int,
                               default=8)


def _initialize_simdim_parser(subparsers):
    simdim_parser = subparsers.add_parser("simdim",
                                          help="Use SimDim for clustering")
    simdim_parser.set_defaults(action="simdim")

    simdim_parser.add_argument("--input",
                               help="gensim model containing embedded entities",
                               type=Path,
                               action=ReadableFile,
                               required=True)
    simdim_parser.add_argument("--output",
                               help="Desired location for storing cluster information",
                               type=Path,
                               action=WriteableDirectory,
                               required=True)
    simdim_parser.add_argument("--output-mode",
                               help=f"Define the type of output. Choose from: {', '.join(VALID_OUTPUT_MODES)}",
                               required=True)
    simdim_parser.add_argument("--threads",
                               help="Number of threads to use",
                               type=int,
                               default=8)


if __name__ == "__main__":
    main()
