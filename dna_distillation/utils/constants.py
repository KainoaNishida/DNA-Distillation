"""Constants and configuration values."""

NUCLEOTIDE_MODELS = {
    "500m": "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "2.5b": "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
}

DOWNSTREAM_TASKS = [
    "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", 
    "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1",
    "promoter_all", "promoter_tata", "promoter_no_tata",
    "enhancers", "enhancers_types", "splice_sites_all",
    "splice_sites_acceptor", "splice_sites_donor",
]
