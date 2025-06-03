from enum import Enum


class UtilityDatasetIdentifier(str, Enum):
    RTE = "rte"
    HELLASWAG = "hellaswag"
    ARC_CHALLANGE = "arc_challenge"
    OEPNBOOKQA = "openbookqa"
    WINOGRANDE = "winogrande"
