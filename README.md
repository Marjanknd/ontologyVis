# RDF / Ontology Viewer

Streamlit app to visualize RDF ontologies (Turtle and JSON-LD) using rdflib + ECharts.

## Features

- Upload `.ttl` or `.json` / `.jsonld` files
- Parse with `rdflib`
- Interactive graph visualization (classes, properties, individuals)
- Summary tables for classes, properties, individuals
- Raw triples view for debugging

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
