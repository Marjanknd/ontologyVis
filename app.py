import io
import json
import chardet
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from rdflib import Graph, RDF, RDFS, OWL, URIRef, Literal, BNode, Namespace

# -----------------------------
# Helpers: RDF parsing & format
# -----------------------------

def detect_rdf_format(filename: str, text: str) -> str:
    """
    Very simple format detection between Turtle and JSON-LD.
    Extend if you want RDF/XML etc.
    """
    if filename:
        lower = filename.lower()
        if lower.endswith((".json", ".jsonld", ".ldjson")):
            return "json-ld"
        if lower.endswith((".ttl", ".turtle")):
            return "turtle"

    # Fallback: inspect content
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return "json-ld"
    # assume Turtle by default
    return "turtle"


def read_file_as_text(uploaded_file) -> str:
    """
    Read uploaded file and return decoded text using chardet.
    """
    raw = uploaded_file.read()
    if not raw:
        return ""
    result = chardet.detect(raw)
    encoding = result.get("encoding") or "utf-8"
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def parse_rdf(text: str, rdf_format: str) -> Graph:
    """
    Parse RDF content (Turtle or JSON-LD) into an rdflib Graph.
    """
    g = Graph()
    g.parse(data=text, format=rdf_format)
    return g


# -----------------------------
# Helpers: ontology analysis
# -----------------------------

def guess_base_namespace(g: Graph) -> str:
    """
    Guess the 'main' namespace by majority vote over URIRefs.
    Useful for hiding external vocabularies (QUDT, DCAT, etc.).
    """
    ns_counts = {}
    for s in g.subjects():
        if isinstance(s, URIRef):
            uri = str(s)
        else:
            continue
        if "#" in uri:
            base = uri.split("#")[0] + "#"
        else:
            # last slash as namespace split
            parts = uri.rsplit("/", 1)
            base = parts[0] + "/" if len(parts) > 1 else uri
        ns_counts[base] = ns_counts.get(base, 0) + 1

    if not ns_counts:
        return ""

    # Most frequent namespace
    base_ns = max(ns_counts.items(), key=lambda x: x[1])[0]
    return base_ns


def get_label(g: Graph, node: URIRef) -> str:
    """
    Get a human-readable label for a node:
    rdfs:label > skos:prefLabel > qname > local fragment.
    """
    from rdflib.namespace import SKOS

    label = g.value(node, RDFS.label)
    if isinstance(label, Literal):
        return str(label)

    pref = g.value(node, SKOS.prefLabel)
    if isinstance(pref, Literal):
        return str(pref)

    # Try qname (prefix:name)
    try:
        return g.qname(node)
    except Exception:
        uri = str(node)
        if "#" in uri:
            return uri.split("#")[-1]
        return uri.rsplit("/", 1)[-1]


def classify_nodes(g: Graph):
    """
    Classify nodes into Classes, ObjectProperties, DatatypeProperties, Ontologies, Individuals.
    """
    classes = set(g.subjects(RDF.type, OWL.Class))
    obj_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
    dt_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    ontologies = set(g.subjects(RDF.type, OWL.Ontology))

    typed_nodes = set()
    for s, p, o in g.triples((None, RDF.type, None)):
        if isinstance(s, URIRef):
            typed_nodes.add(s)

    # Individuals = subjects that have rdf:type but are not in the above sets
    individuals = {s for s in typed_nodes if s not in classes
                   and s not in obj_props and s not in dt_props and s not in ontologies}

    return classes, obj_props, dt_props, ontologies, individuals


def extract_summary_tables(g: Graph, classes, obj_props, dt_props, individuals):
    """
    Build small DataFrames for classes, properties, individuals.
    """
    def build_df(nodes):
        data = [{"IRI": str(n), "Label": get_label(g, n)} for n in sorted(nodes, key=lambda x: get_label(g, x).lower())]
        return pd.DataFrame(data)

    return {
        "classes": build_df(classes),
        "obj_props": build_df(obj_props),
        "dt_props": build_df(dt_props),
        "individuals": build_df(individuals),
    }


# -----------------------------
# Graph -> ECharts conversion
# -----------------------------

def build_echarts_graph(
    g: Graph,
    hide_external: bool = True,
    show_schema_edges: bool = True,
    show_object_property_edges: bool = True,
    show_type_edges: bool = False,
):
    """
    Convert rdflib.Graph into nodes & links for ECharts.

    - Hides external vocabularies if hide_external=True
    - Only includes URI nodes (skips blank nodes & literals)
    """
    base_ns = guess_base_namespace(g)

    classes, obj_props, dt_props, ontologies, individuals = classify_nodes(g)

    # Node categories & colors
    categories = {
        "Ontology": "#9b59b6",
        "Class": "#5470c6",
        "ObjectProperty": "#ee6666",
        "DatatypeProperty": "#91cc75",
        "Individual": "#3ba272",
        "External": "#a5a5a5",
        "Other": "#ccc",
    }

    def get_category(node: URIRef) -> str:
        if node in ontologies:
            return "Ontology"
        if node in classes:
            return "Class"
        if node in obj_props:
            return "ObjectProperty"
        if node in dt_props:
            return "DatatypeProperty"
        if node in individuals:
            return "Individual"

        # external / other
        uri = str(node)
        if base_ns and not uri.startswith(base_ns):
            return "External"
        return "Other"

    # Build nodes
    nodes = []
    node_ids = set()

    def add_node(node: URIRef):
        if not isinstance(node, URIRef):
            return
        uri = str(node)

        # Optional external filtering
        if hide_external and base_ns and not uri.startswith(base_ns):
            return

        if uri in node_ids:
            return

        cat = get_category(node)
        label = get_label(g, node)
        size_map = {
            "Ontology": 26,
            "Class": 22,
            "ObjectProperty": 18,
            "DatatypeProperty": 18,
            "Individual": 14,
            "External": 12,
            "Other": 10,
        }
        symbol_size = size_map.get(cat, 10)

        node_obj = {
            "id": uri,              # internal ID
            "name": label,          # what is shown
            "value": uri,           # full IRI in tooltip
            "symbolSize": symbol_size,
            "category": cat,
            "itemStyle": {"color": categories.get(cat, "#ccc")},
            "label": {
                "show": True,
                "formatter": label,
                "overflow": "truncate"
            }
        }
        node_ids.add(uri)
        nodes.append(node_obj)

    # First, add all "important" nodes
    important_nodes = set().union(classes, obj_props, dt_props, ontologies, individuals)
    for n in important_nodes:
        add_node(n)

    # Build edges
    links = []

    def add_edge(s: URIRef, p: URIRef, o: URIRef, label: str):
        """
        Add edge (s -> o) with label.
        """
        if hide_external and base_ns:
            if not str(s).startswith(base_ns) and not str(o).startswith(base_ns):
                # drop external–external links
                return

        add_node(s)
        add_node(o)

        if str(s) not in node_ids or str(o) not in node_ids:
            return

        links.append({
            "source": str(s),
            "target": str(o),
            "value": label
        })

    # Pre-calc predicates that are ObjectProperties
    object_property_nodes = obj_props.copy()

    # Iterate triples for edges
    for s, p, o in g.triples((None, None, None)):
        # skip blank nodes and literals as nodes
        if not isinstance(s, URIRef):
            continue

        # TYPE edges: s --rdf:type--> o
        if p == RDF.type:
            if isinstance(o, URIRef) and show_type_edges:
                try:
                    label = g.qname(p)
                except Exception:
                    label = "rdf:type"
                add_edge(s, p, o, label)
            continue

        # range & domain & subclass & inverseOf edges
        if isinstance(o, URIRef):
            if p in (RDFS.subClassOf, RDFS.domain, RDFS.range, OWL.inverseOf):
                if not show_schema_edges:
                    continue
                try:
                    label = g.qname(p)
                except Exception:
                    label = str(p).split("#")[-1]
                add_edge(s, p, o, label)
                continue

            # object property links (measurement -> hasParameter -> Parameter)
            if p in object_property_nodes and show_object_property_edges:
                try:
                    label = g.qname(p)
                except Exception:
                    label = str(p).split("#")[-1]
                add_edge(s, p, o, label)
                continue

        # Ignore all other literals and bnodes for the graph
        # (they are visible from the RDF triples table if needed)
        # You can extend here if you want to visualize quantityKind/unit, etc.

    return nodes, links, categories


def create_echarts_option(nodes, links, categories, layout="force"):
    """
    Build ECharts option dict for Streamlit-ECharts.
    """
    category_list = [{"name": name, "itemStyle": {"color": color}}
                     for name, color in categories.items()]

    option = {
        "title": {
            "text": "Ontology / RDF Graph",
            "subtext": f"{layout.capitalize()} layout",
            "top": "top",
            "left": "center"
        },
        "tooltip": {
            "show": True,
            "trigger": "item",
            "formatter": "{b}<br />{c}"
        },
        "legend": [{
            "data": list(categories.keys()),
            "orient": "vertical",
            "left": "left",
            "top": "middle"
        }],
        "animationDurationUpdate": 1500,
        "animationEasingUpdate": "quinticInOut",
        "series": [{
            "name": "Ontology Graph",
            "type": "graph",
            "layout": layout,
            "data": nodes,
            "links": links,
            "categories": category_list,
            "roam": True,
            "draggable": True,
            "focusNodeAdjacency": True,
            "label": {
                "position": "right",
                "formatter": "{b}",
                "show": True
            },
            "lineStyle": {
                "width": 1,
                "curveness": 0.2
            },
            "edgeSymbol": ["none", "arrow"],
            "edgeSymbolSize": [4, 10],
            "edgeLabel": {
                "show": True,
                "fontSize": 8,
                "formatter": "{c}"
            },
            "emphasis": {
                "focus": "adjacency",
                "lineStyle": {"width": 3}
            }
        }]
    }
    return option


# -----------------------------
# Streamlit app
# -----------------------------

def main():
    st.set_page_config(page_title="Ontology / RDF Graph Viewer", layout="wide")
    st.title("Ontology / RDF Graph Viewer")

    st.markdown(
        "Upload a **Turtle (.ttl)** or **JSON-LD (.json / .jsonld)** file. "
        "The app will parse it with rdflib and show an interactive graph + summary."
    )

    st.sidebar.header("Upload & Options")
    uploaded_file = st.sidebar.file_uploader(
        "Upload ontology (TTL or JSON-LD)",
        type=["ttl", "turtle", "json", "jsonld"]
    )

    layout = st.sidebar.selectbox("Graph layout", ["force", "circular"])
    hide_external = st.sidebar.checkbox(
        "Hide external vocabularies (QUDT, DCAT, PROV, etc.)",
        value=True
    )
    show_schema_edges = st.sidebar.checkbox(
        "Show schema edges (subClassOf, domain, range, inverseOf)",
        value=True
    )
    show_obj_edges = st.sidebar.checkbox(
        "Show object-property edges (hasParameter, measures, ...)",
        value=True
    )
    show_type_edges = st.sidebar.checkbox(
        "Show rdf:type edges",
        value=False
    )

    if uploaded_file is None:
        st.info("Upload your hydrogen ontology TTL or JSON-LD file in the sidebar to get started.")
        return

    # Read and parse RDF
    try:
        text = read_file_as_text(uploaded_file)
        rdf_format = detect_rdf_format(uploaded_file.name, text)
        g = parse_rdf(text, rdf_format)
    except Exception as e:
        st.error(f"Error parsing RDF file: {e}")
        return

    st.success(f"Parsed ontology as **{rdf_format}** with **{len(g)}** triples.")

    classes, obj_props, dt_props, ontologies, individuals = classify_nodes(g)
    summary_tables = extract_summary_tables(g, classes, obj_props, dt_props, individuals)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Triples", len(g))
    col2.metric("Classes", len(classes))
    col3.metric("Object properties", len(obj_props))
    col4.metric("Datatype properties", len(dt_props))
    col5.metric("Individuals", len(individuals))

    tabs = st.tabs(["Graph", "Summary", "Triples", "Raw text"])

    # --- Graph tab ---
    with tabs[0]:
        nodes, links, categories = build_echarts_graph(
            g,
            hide_external=hide_external,
            show_schema_edges=show_schema_edges,
            show_object_property_edges=show_obj_edges,
            show_type_edges=show_type_edges,
        )

        if not nodes or not links:
            st.error("No nodes or links extracted for visualization. "
                     "Try enabling more edge types or turning off 'Hide external vocabularies'.")
        else:
            option = create_echarts_option(nodes, links, categories, layout=layout)
            st_echarts(options=option, height="900px")

    # --- Summary tab ---
    with tabs[1]:
        st.subheader("Classes")
        st.dataframe(summary_tables["classes"], use_container_width=True, hide_index=True)

        st.subheader("Object properties")
        st.dataframe(summary_tables["obj_props"], use_container_width=True, hide_index=True)

        st.subheader("Datatype properties")
        st.dataframe(summary_tables["dt_props"], use_container_width=True, hide_index=True)

        st.subheader("Individuals (instances of Measurement etc.)")
        st.dataframe(summary_tables["individuals"], use_container_width=True, hide_index=True)

    # --- Triples tab ---
    with tabs[2]:
        st.subheader("RDF triples (first 500)")
        data = []
        for i, (s, p, o) in enumerate(g):
            if i >= 500:
                break
            data.append({
                "subject": str(s),
                "predicate": str(p),
                "object": str(o),
                "object_is_literal": isinstance(o, Literal)
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    # --- Raw text tab ---
    with tabs[3]:
        st.subheader("Raw file content")
        st.code(text, language="turtle" if rdf_format == "turtle" else "json")


if __name__ == "__main__":
    main()
