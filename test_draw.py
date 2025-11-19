
import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_drawio_architecture(filename="database_architecture.drawio"):
    """
    生成一个可被 draw.io (diagrams.net) 直接打开/导出的 .drawio 文件。
    关键修复：为 mxGeometry 使用 attrib={'as': 'geometry'}，而不是 as_，否则会生成 as_ 属性导致无效。
    """

    # 根节点（可适当带上版本信息，非必需）
    mxfile = ET.Element(
        "mxfile",
        host="app.diagrams.net",
        modified="2025-11-18T00:00:00Z",
        agent="python-xml.etree",
        version="20.8.23",
    )

    # diagram 节点（id 可选，但建议提供）
    diagram = ET.SubElement(mxfile, "diagram", id="diagram-1", name="一体化数据库架构")

    # mxGraphModel 可带一些默认配置（非强制）
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        dx="1360",
        dy="768",
        grid="1",
        gridSize="10",
        guides="1",
        tooltips="1",
        connect="1",
        arrows="1",
        fold="1",
        page="1",
        pageScale="1",
        pageWidth="827",
        pageHeight="1169",
        math="0",
        shadow="0",
    )

    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    def add_node(id_, text, x, y, color="#dae8fc", width=180, height=60):
        """
        添加一个矩形节点（vertex）
        注意：mxGeometry 必须使用 attrib={'as': 'geometry'}
        """
        cell = ET.SubElement(
            root,
            "mxCell",
            id=str(id_),
            value=text,
            style=f"rounded=1;whiteSpace=wrap;html=1;fillColor={color};",
            vertex="1",
            parent="1",
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            attrib={
                "x": str(x),
                "y": str(y),
                "width": str(width),
                "height": str(height),
                "as": "geometry",
            },
        )
        return cell

    def add_edge(id_, src, tgt, label=None, style=None):
        """
        添加一条边（edge），默认正交线 + 箭头
        """
        edge_style = (
            style
            or "edgeStyle=orthogonalEdgeStyle;rounded=1;endArrow=classic;html=1;"
        )
        edge = ET.SubElement(
            root,
            "mxCell",
            id=str(id_),
            value=(label or ""),
            style=edge_style,
            edge="1",
            parent="1",
            source=str(src),
            target=str(tgt),
        )
        ET.SubElement(edge, "mxGeometry", attrib={"relative": "1", "as": "geometry"})
        return edge

    # 示例：三层结构
    add_node(2, "一体化产品层", 100, 50, "#dae8fc")
    add_node(3, "一体化引擎层", 100, 150, "#fff2cc")
    add_node(4, "一体化架构层", 100, 250, "#d5e8d4")

    add_edge(5, 2, 3)
    add_edge(6, 3, 4)

    # 美化输出（可选）
    raw = ET.tostring(mxfile, encoding="utf-8")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ", encoding="utf-8")

    with open(filename, "wb") as f:
        f.write(pretty)

    print(f"✅ Draw.io 文件已生成: {filename}")


if __name__ == "__main__":
    create_drawio_architecture()
