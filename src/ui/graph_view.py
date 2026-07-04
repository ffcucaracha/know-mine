from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


def render_graph(edges: pd.DataFrame) -> bool:
    if edges.empty:
        st.info("Для выбранной сущности связи не найдены.")
        return True

    try:
        network = Network(height="560px", width="100%", directed=True)
        network.barnes_hut()
        network.set_options(
            """
            {
              "nodes": {
                "shape": "dot",
                "size": 18,
                "font": {
                  "size": 18,
                  "face": "Arial",
                  "color": "#222222",
                  "strokeWidth": 4,
                  "strokeColor": "#ffffff",
                  "multi": "html"
                },
                "borderWidth": 2
              },
              "edges": {
                "arrows": {
                  "to": {
                    "enabled": true,
                    "scaleFactor": 0.7
                  }
                },
                "font": {
                  "size": 13,
                  "align": "middle",
                  "color": "#555555",
                  "strokeWidth": 3,
                  "strokeColor": "#ffffff"
                },
                "smooth": {
                  "type": "dynamic"
                }
              },
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -26000,
                  "centralGravity": 0.25,
                  "springLength": 160,
                  "springConstant": 0.04,
                  "damping": 0.12
                },
                "minVelocity": 0.75
              },
              "interaction": {
                "hover": true,
                "tooltipDelay": 120,
                "hideEdgesOnDrag": false,
                "navigationButtons": true,
                "keyboard": true
              }
            }
            """
        )

        for edge in edges.to_dict("records"):
            source_id = str(edge["source_node_id"])
            target_id = str(edge["target_node_id"])
            source_label = str(edge["source_label"])
            target_label = str(edge["target_label"])
            source_type = str(edge["source_type"])
            target_type = str(edge["target_type"])
            relation = str(edge["relation"])

            network.add_node(
                source_id,
                label=f"{source_label}\n({source_type})",
                title=f"{source_label} | {source_type}",
            )
            network.add_node(
                target_id,
                label=f"{target_label}\n({target_type})",
                title=f"{target_label} | {target_type}",
            )
            network.add_edge(
                source_id,
                target_id,
                label=relation,
                title=str(edge.get("evidence") or relation),
            )

        html = network.generate_html()
        components.html(html, height=590, scrolling=True)
        st.caption(
            "Управление картой: перетаскивайте узлы мышью, колесом меняйте масштаб, "
            "зажмите и тяните фон для перемещения карты, наведите курсор на узел или "
            "связь для подробностей."
        )
        return True
    except Exception as exc:
        st.warning(f"Граф не удалось построить, показываю таблицу связей: {exc}")
        return False
