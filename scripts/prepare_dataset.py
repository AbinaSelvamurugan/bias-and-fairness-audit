#!/usr/bin/env python3
"""Validate and display dataset/source summary without publishing row-level data."""
import json
from fairaudit.config import load_configs
from fairaudit.data.loader import load_dataset,provenance
from fairaudit.data.validation import validate
f=load_dataset(); c=load_configs()
s=validate(f,c)
print(json.dumps({'provenance':provenance(f),'quality':s},indent=2,default=str))
