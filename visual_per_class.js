// visual_per_class.js

// 配色方案（红-橙-绿）
function accuracyColor(val) {
    // 红->橙->绿插值，val应在[0,1]范围
    let c = d3.scaleLinear()
        .domain([0, 0.5, 1])
        .range(["#FF2222", "#FFA500", "#4FC36E"]);
    return c(val);
}

/**
 * 渲染 per-class top-k accuracy 热力表
 * @param {Array} data  CSV读取后的对象数组
 * @param {String} containerSelector 例如 "#output"
 */
function renderPerClassHeatmap(data, containerSelector) {
    // 找到所有Top-K Acc列
    let cols = Object.keys(data[0]).filter(c => c.startsWith("Top-") && c.endsWith("Acc"));

    // 清理原内容（可选）
    d3.select(containerSelector).html("");

    // 构建表格
    let table = d3.select(containerSelector).append("table");
    let thead = table.append("thead");
    let tbody = table.append("tbody");

    // 表头
    thead.append("tr")
        .selectAll("th")
        .data(["Class"].concat(cols))
        .enter()
        .append("th")
        .text(d => d);

    // 表体
    let rows = tbody.selectAll("tr")
        .data(data)
        .enter()
        .append("tr");

    rows.append("td").text(d => d["Class"]);

    cols.forEach(col => {
        rows.append("td")
            .text(d => Number(d[col]).toFixed(2))
            .style("background", d => accuracyColor(Number(d[col])))
            .style("color", d => Number(d[col]) > 0.6 ? "#fff" : "#222");
    });
}

// 导出功能给html用
window.renderPerClassHeatmap = renderPerClassHeatmap;
