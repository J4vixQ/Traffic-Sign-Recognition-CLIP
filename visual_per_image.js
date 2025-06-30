// 配色方案（绿色渐变/红色渐变）
function makeColor(prob, correct) {
    // prob in [0, 1]
    // correct: true => green; false => red
    let color;
    if (correct) {
        color = d3.interpolateRgb("#FFFFFF", "#22BB22")(prob); // 白到绿
    } else {
        color = d3.interpolateRgb("#FFFFFF", "#FF2222")(prob); // 白到红
    }
    return color;
}

const LABEL_ABBR = {
  "animal crossing": "anml",
  "bumpy road": "bp",
  "cross road": "crs",
  "cycle crossing": "cc",
  "dip": "dip",
  "gas station": "gas",
  "give way": "gw",
  "height limit": "hl",
  "hospital": "hp",
  "left turn": "lt",
  "narrow roads ahead": "nrw",
  "no entry for vehicles": "novc",
  "no overtaking": "noot",
  "no parking": "nopk",
  "no u turn": "nout",
  "pedestrian crossing": "pd",
  "railway crossing": "rl",
  "right turn": "rt",
  "road work ahead": "rw",
  "round about": "rnd",
  "school zone": "sch",
  "slippery road": "slp",
  "speed limit": "spd",
  "steep ascent": "asc",
  "steep descent": "des",
  "stop": "stp",
  "weight limit": "wtl",
  "width limit": "wdl"
};


/**
 * 渲染 per-image top-k label 热力表
 * @param {Array} data  CSV读取后的对象数组
 * @param {String} containerSelector 例如 "#output"
 */
function renderPerImageHeatmap(data, containerSelector) {
    const K = 5;  // 只显示Top1~Top3
    const labelCols = Array.from({length: K}, (_,i) => `Top${i+1}_Label`);
    const probCols  = Array.from({length: K}, (_,i) => `Top${i+1}_Prob`);
    const labelHeaders = Array.from({length: K}, (_,i) => `Label ${i+1}`);

    d3.select(containerSelector).html("");

    let table = d3.select(containerSelector).append("table");
    let thead = table.append("thead");
    let tbody = table.append("tbody");

    // 表头
    thead.append("tr")
        .selectAll("th")
        .data(["Image"].concat(labelHeaders))
        .enter()
        .append("th")
        .text(d => d);

    // 表体
    let rows = tbody.selectAll("tr")
        .data(data)
        .enter()
        .append("tr");

    // Image 列：去除 .png，下划线替换空格
    rows.append("td")
        .text(d => d["Image"].replace(/\.png$/,"").replace(/_/g, " "))
        .attr("title", d => d["Image"]);

    // Top1~Top3预测列
    for (let i = 0; i < K; ++i) {
        rows.append("td")
            .text(d => {
                let orig = d[labelCols[i]] || "";
                return LABEL_ABBR[orig] || orig.slice(0,4);
            })
            .attr("title", d => d[labelCols[i]])
            .style("background", d => {
                let image = d["Image"];
                let trueLabel = image.replace(/\d+\.png$/, "")
                                .replace(/[_-]+$/, "")
                                .replace(/_/g, " ")
                                .trim();
                let correct = (d[labelCols[i]] === trueLabel);
                let prob = Number(d[probCols[i]]) || 0.0;
                return makeColor(prob, correct);
            })
            .style("color", d => {
                let prob = Number(d[probCols[i]]) || 0.0;
                let c = d3.color(makeColor(prob, true));
                let brightness = 0.299*c.r/255 + 0.587*c.g/255 + 0.114*c.b/255;
                return brightness > 0.65 ? "#222" : "#fff";
            });
    }

}



// 导出
window.renderPerImageHeatmap = renderPerImageHeatmap;
