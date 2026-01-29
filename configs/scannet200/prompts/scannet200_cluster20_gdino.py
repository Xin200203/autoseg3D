# 20-cluster prompt for category-aware GDINO object queries (ScanNet200 SV).
#
# NOTE: Keep each concept as a short noun phrase and end with '.' so that
# GroundingDINO treats them as separate text queries.
GDINO_CLUSTER20_PHRASES = [
    "chair",
    "sofa",
    "stool",
    "bench",
    "table",
    "desk",
    "counter",
    "cabinet",
    "shelf",
    "bookshelf",
    "wardrobe",
    "bed",
    "pillow",
    "blanket",
    "toilet",
    "sink",
    "bathtub",
    "lamp",
    "tv",
    "monitor",
]

GDINO_CLUSTER20_CAPTION = ". ".join(GDINO_CLUSTER20_PHRASES) + "."
