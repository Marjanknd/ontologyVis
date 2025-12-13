import chardet
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from rdflib import Graph, RDF, RDFS, OWL, URIRef, Literal, BNode

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

    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return "json-ld"
    return "turtle"


def read_file_as_text(uploaded_file) -> str:
    """Read uploaded file and return decoded text using chardet."""
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
    """Parse RDF content (Turtle or JSON-LD) into an rdflib Graph."""
    g = Graph()
    g.parse(data=text, format=rdf_format)
    return g


# -----------------------------
# Helpers: ontology analysis
# -----------------------------

def guess_base_namespace(g: Graph) -> str:
    """
    Guess the 'main' namespace by majority vote over URIRefs subjects.
    Useful for hiding external vocabularies (QUDT, DCAT, etc.).
    """
    ns_counts: dict[str, int] = {}
    for s in g.subjects():
        if not isinstance(s, URIRef):
            continue
        uri = str(s)
        if "#" in uri:
            base = uri.split("#")[0] + "#"
        else:
            parts = uri.rsplit("/", 1)
            base = parts[0] + "/" if len(parts) > 1 else uri
        ns_counts[base] = ns_counts.get(base, 0) + 1

    if not ns_counts:
        return ""
    return max(ns_counts.items(), key=lambda x: x[1])[0]


def get_label(g: Graph, node: URIRef) -> str:
    """
    Get a human-readable label for a URI node:
    rdfs:label > skos:prefLabel > qname > local fragment.
    """
    from rdflib.namespace import SKOS

    label = g.value(node, RDFS.label)
    if isinstance(label, Literal):
        return str(label)

    pref = g.value(node, SKOS.prefLabel)
    if isinstance(pref, Literal):
        return str(pref)

    try:
        return g.qname(node)
    except Exception:
        uri = str(node)
        if "#" in uri:
            return uri.split("#")[-1]
        return uri.rsplit("/", 1)[-1]


def classify_nodes(g: Graph):
    """Classify nodes into Classes, ObjectProperties, DatatypeProperties, Ontologies, Individuals."""
    classes = set(g.subjects(RDF.type, OWL.Class))
    obj_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
    dt_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    ontologies = set(g.subjects(RDF.type, OWL.Ontology))

    typed_nodes = set()
    for s, _, _ in g.triples((None, RDF.type, None)):
        if isinstance(s, URIRef):
            typed_nodes.add(s)

    individuals = {
        s for s in typed_nodes
        if s not in classes and s not in obj_props and s not in dt_props and s not in ontologies
    }
    return classes, obj_props, dt_props, ontologies, individuals


def extract_summary_tables(g: Graph, classes, obj_props, dt_props, individuals):
    """Build small DataFrames for classes, properties, individuals."""
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
    include_bnodes: bool = False,
):
    """
    Convert rdflib.Graph into nodes & links for ECharts.

    - Hides external vocabularies if hide_external=True (URIRef only)
    - Optionally includes BNodes if include_bnodes=True
    - Ignores literals as nodes (always)
    """
    base_ns = guess_base_namespace(g)
    classes, obj_props, dt_props, ontologies, individuals = classify_nodes(g)

    categories = {
        "Ontology": "#9b59b6",
        "Class": "#5470c6",
        "ObjectProperty": "#ee6666",
        "DatatypeProperty": "#91cc75",
        "Individual": "#3ba272",
        "BlankNode": "#888888",
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
        uri = str(node)
        if base_ns and not uri.startswith(base_ns):
            return "External"
        return "Other"

    nodes = []
    node_ids = set()

    def node_id(n):
        if isinstance(n, URIRef):
            return str(n)
        if include_bnodes and isinstance(n, BNode):
            return "_:" + str(n)
        return None

    def add_node(node):
        nid = node_id(node)
        if nid is None or nid in node_ids:
            return

        # External filtering applies only to URIRefs
        if isinstance(node, URIRef):
            uri = str(node)
            if hide_external and base_ns and not uri.startswith(base_ns):
                return
            cat = get_category(node)
            label = get_label(g, node)
            tooltip_value = uri
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
        else:
            # BNode
            cat = "BlankNode"
            label = "blank:" + str(node)[:8]
            tooltip_value = nid
            symbol_size = 10

        node_ids.add(nid)
        nodes.append({
            "id": nid,
            "name": label,
            "value": tooltip_value,
            "symbolSize": symbol_size,
            "category": cat,
            "itemStyle": {"color": categories.get(cat, "#ccc")},
            "label": {"show": True, "formatter": label, "overflow": "truncate"},
        })

    # Seed nodes: all "important" URIRefs
    important_nodes = set().union(classes, obj_props, dt_props, ontologies, individuals)
    for n in important_nodes:
        add_node(n)

    links = []

    def add_edge(s, p, o, label: str):
        # If hide_external: drop external–external URIRef links
        if hide_external and base_ns:
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                if not str(s).startswith(base_ns) and not str(o).startswith(base_ns):
                    return

        add_node(s)
        add_node(o)

        sid, oid = node_id(s), node_id(o)
        if sid is None or oid is None:
            return
        if sid not in node_ids or oid not in node_ids:
            return

        links.append({"source": sid, "target": oid, "value": label})

    object_property_nodes = obj_props.copy()

    for s, p, o in g.triples((None, None, None)):
        # Keep original behavior: skip BNode subjects
        if not isinstance(s, URIRef):
            continue

        # rdf:type edges
        if p == RDF.type:
            if show_type_edges and isinstance(o, URIRef):
                try:
                    lbl = g.qname(p)
                except Exception:
                    lbl = "rdf:type"
                add_edge(s, p, o, lbl)
            continue

        # schema edges (only URIRef objects)
        if isinstance(o, URIRef) and p in (RDFS.subClassOf, RDFS.domain, RDFS.range, OWL.inverseOf):
            if not show_schema_edges:
                continue
            try:
                lbl = g.qname(p)
            except Exception:
                lbl = str(p).split("#")[-1]
            add_edge(s, p, o, lbl)
            continue

        # object property edges (URIRef or BNode object if enabled)
        if p in object_property_nodes and show_object_property_edges:
            if isinstance(o, URIRef) or (include_bnodes and isinstance(o, BNode)):
                try:
                    lbl = g.qname(p)
                except Exception:
                    lbl = str(p).split("#")[-1]
                add_edge(s, p, o, lbl)
            continue

        # ignore literals + other predicates for visualization

    return nodes, links, categories


def create_echarts_option(nodes, links, categories, layout="force", title="Ontology / RDF Graph"):
    category_list = [{"name": name, "itemStyle": {"color": color}} for name, color in categories.items()]
    return {
        "title": {"text": title, "subtext": f"{layout.capitalize()} layout", "top": "top", "left": "center"},
        "tooltip": {"show": True, "trigger": "item", "formatter": "{b}<br />{c}"},
        "legend": [{"data": list(categories.keys()), "orient": "vertical", "left": "left", "top": "middle"}],
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
            "label": {"position": "right", "formatter": "{b}", "show": True},
            "lineStyle": {"width": 1, "curveness": 0.2},
            "edgeSymbol": ["none", "arrow"],
            "edgeSymbolSize": [4, 10],
            "edgeLabel": {"show": True, "fontSize": 14, "formatter": "{c}", "backgroundColor": "rgba(255,255,255,0.75)","padding": [2, 4, 2, 4],"borderRadius": 3},
            "emphasis": {"focus": "adjacency", "lineStyle": {"width": 3}},
        }]
    }


# -----------------------------
# Search (filter subgraph)
# -----------------------------

def filter_graph_by_query(
    nodes: list[dict],
    links: list[dict],
    query: str,
    include_neighbors: bool = True,
    include_links_among_results: bool = True,
    max_nodes: int = 400,
):
    """
    Filter nodes/links to only what matches `query`, plus (optionally) 1-hop neighbors.
    Matching is done against node['name'] (label) and node['id'] (IRI or _:bnode).
    """
    q = (query or "").strip().lower()
    if not q:
        return [], [], [], pd.DataFrame()

    # Index nodes
    node_by_id = {n["id"]: n for n in nodes}
    def matches(n: dict) -> bool:
        return q in str(n.get("name", "")).lower() or q in str(n.get("id", "")).lower() or q in str(n.get("value", "")).lower()

    matched_ids = {nid for nid, n in node_by_id.items() if matches(n)}
    if not matched_ids:
        return [], [], [], pd.DataFrame()

    keep_ids = set(matched_ids)
    keep_links = []

    # Edges touching matches -> keep (and optionally include neighbors)
    for e in links:
        s, t = e["source"], e["target"]
        if s in matched_ids or t in matched_ids:
            keep_links.append(e)
            if include_neighbors:
                keep_ids.add(s)
                keep_ids.add(t)

    # Optionally include additional links among kept nodes (context)
    if include_links_among_results:
        for e in links:
            s, t = e["source"], e["target"]
            if s in keep_ids and t in keep_ids:
                keep_links.append(e)

    # De-duplicate links
    seen = set()
    uniq_links = []
    for e in keep_links:
        key = (e["source"], e["target"], e.get("value", ""))
        if key in seen:
            continue
        seen.add(key)
        uniq_links.append(e)

    # Build filtered node list (cap max_nodes)
    keep_ids_list = list(keep_ids)
    if len(keep_ids_list) > max_nodes:
        # Prioritize matched nodes first, then others
        ordered = list(matched_ids) + [nid for nid in keep_ids_list if nid not in matched_ids]
        keep_ids = set(ordered[:max_nodes])

        # Also drop links that now reference removed nodes
        uniq_links = [e for e in uniq_links if e["source"] in keep_ids and e["target"] in keep_ids]

    filtered_nodes = [node_by_id[nid] for nid in keep_ids if nid in node_by_id]

    # A small table for matched nodes
    matched_rows = []
    for nid in sorted(matched_ids):
        n = node_by_id.get(nid)
        if not n:
            continue
        matched_rows.append({
            "Label": n.get("name", ""),
            "ID": n.get("id", ""),
            "Category": n.get("category", ""),
        })
    matched_df = pd.DataFrame(matched_rows)

    return filtered_nodes, uniq_links, sorted(matched_ids), matched_df


# -----------------------------
# Streamlit app
# -----------------------------

def main():
    st.set_page_config(page_title="Ontology / RDF Graph Viewer", layout="wide")
    st.title("Ontology / RDF Graph Viewer")

    st.markdown(
        "Upload a **Turtle (.ttl)** or **JSON-LD (.json / .jsonld)** file. "
        "The app parses with `rdflib` and shows an interactive graph + summaries."
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
    include_bnodes = st.sidebar.checkbox(
        "Include blank nodes (BNodes) in visualization",
        value=False
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
        st.info("Upload your ontology TTL or JSON-LD file in the sidebar to get started.")
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

    tabs = st.tabs(["Graph", "Search", "Summary", "Triples", "Raw text"])

    # Build the full visualization graph once (then Search tab can filter it)
    full_nodes, full_links, full_categories = build_echarts_graph(
        g,
        hide_external=hide_external,
        show_schema_edges=show_schema_edges,
        show_object_property_edges=show_obj_edges,
        show_type_edges=show_type_edges,
        include_bnodes=include_bnodes,
    )

    # --- Graph tab (full graph) ---
    with tabs[0]:
        if not full_nodes or not full_links:
            st.error(
                "No nodes or links extracted for visualization. "
                "Try enabling more edge types or turning off 'Hide external vocabularies'."
            )
        else:
            option = create_echarts_option(full_nodes, full_links, full_categories, layout=layout, title="Ontology / RDF Graph (Full)")
            st_echarts(options=option, height="900px")

    # --- Search tab (filtered subgraph) ---
    with tabs[1]:
        st.subheader("Search nodes & relationships")
        query = st.text_input("Search (matches label / qname / IRI / blank-node id)", value="", placeholder="e.g., Parameter, hasQuantityValue, Measurement, temperature...")

        c1, c2, c3 = st.columns([1, 1, 1])
        include_neighbors = c1.checkbox("Include 1-hop neighbors", value=True)
        include_links_among = c2.checkbox("Include links among results", value=True)
        max_nodes = c3.number_input("Max nodes", min_value=50, max_value=2000, value=400, step=50)

        if not query.strip():
            st.info("Type a search term to filter the graph and show only relevant nodes + relationships.")
        else:
            f_nodes, f_links, matched_ids, matched_df = filter_graph_by_query(
                full_nodes,
                full_links,
                query=query,
                include_neighbors=include_neighbors,
                include_links_among_results=include_links_among,
                max_nodes=int(max_nodes),
            )

            if not f_nodes:
                st.warning("No matches found. Try a different keyword.")
            else:
                st.caption(f"Matched nodes: **{len(matched_ids)}** | Displayed nodes: **{len(f_nodes)}** | Displayed edges: **{len(f_links)}**")
                option = create_echarts_option(f_nodes, f_links, full_categories, layout=layout, title=f"Search view: {query}")
                st_echarts(options=option, height="900px")

                with st.expander("Matched nodes (table)"):
                    st.dataframe(matched_df, use_container_width=True, hide_index=True)

                with st.expander("Edges in this view (table)"):
                    edge_rows = [{"source": e["source"], "predicate": e.get("value", ""), "target": e["target"]} for e in f_links]
                    st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)

    # --- Summary tab ---
    with tabs[2]:
        st.subheader("Classes")
        st.dataframe(summary_tables["classes"], use_container_width=True, hide_index=True)

        st.subheader("Object properties")
        st.dataframe(summary_tables["obj_props"], use_container_width=True, hide_index=True)

        st.subheader("Datatype properties")
        st.dataframe(summary_tables["dt_props"], use_container_width=True, hide_index=True)

        st.subheader("Individuals")
        st.dataframe(summary_tables["individuals"], use_container_width=True, hide_index=True)

    # --- Triples tab ---
    with tabs[3]:
        st.subheader("RDF triples (first 500)")
        data = []
        for i, (s, p, o) in enumerate(g):
            if i >= 500:
                break
            data.append({
                "subject": str(s),
                "predicate": str(p),
                "object": str(o),
                "object_is_literal": isinstance(o, Literal),
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    # --- Raw text tab ---
    with tabs[4]:
        st.subheader("Raw file content")
        st.code(text, language="turtle" if rdf_format == "turtle" else "json")


if __name__ == "__main__":
    main()
