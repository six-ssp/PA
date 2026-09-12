/**
 * 使用 @oai/artifact-tool 把 Python 数值结果写入四个官方 Excel 模板。
 * 运行前先执行 `python src/run_all.py` 生成 intermediate/*.json。
 */
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

// 以脚本位置定位仓库，允许从其他工作目录执行。
const root = path.dirname(fileURLToPath(import.meta.url));
const inputDir = path.join(root, "附件3");
const intermediateDir = path.join(root, "intermediate");
const outputDir = path.join(root, "results");

async function loadJson(name) {
  try {
    return JSON.parse(await fs.readFile(path.join(intermediateDir, name), "utf8"));
  } catch (error) {
    throw new Error(`无法读取 intermediate/${name}；请先执行 python src/run_all.py`, { cause: error });
  }
}

function validatePayload(payload, fields, name) {
  if (!Array.isArray(payload.time) || !payload.time.length || !Array.isArray(payload.distance)) {
    throw new Error(`${name}: 缺少时间或距离数组`);
  }
  if (payload.time[0] !== 0 || payload.time.some((v, i, a) => !Number.isFinite(v) || (i > 0 && v <= a[i - 1]))) {
    throw new Error(`${name}: 时间必须从 0 开始并严格递增`);
  }
  for (const field of fields) {
    const matrix = payload[field];
    if (!Array.isArray(matrix) || matrix.length !== payload.time.length ||
        matrix.some(row => !Array.isArray(row) || row.length !== payload.distance.length ||
          row.some(value => value !== null && (typeof value !== "number" || !Number.isFinite(value))))) {
      throw new Error(`${name}/${field}: 矩阵尺寸或数值非法`);
    }
  }
}

async function importTemplate(name) {
  const blob = await FileBlob.load(path.join(inputDir, name));
  return SpreadsheetFile.importXlsx(blob);
}

function writeMatrix(sheet, times, distances, values) {
  const rows = values.length + 1;
  const cols = distances.length + 1;
  const matrix = [["时间\\到药材中心的距离", ...distances]];
  for (let i = 0; i < values.length; i += 1) {
    matrix.push([times[i], ...values[i]]);
  }
  sheet.getRangeByIndexes(0, 0, rows, cols).values = matrix;
  const used = sheet.getRangeByIndexes(0, 0, rows, cols);
  used.format.font = { name: "Arial", size: 10 };
  used.format.verticalAlignment = "center";
  sheet.getRangeByIndexes(0, 0, 1, cols).format = {
    fill: "#1F4E78",
    font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRangeByIndexes(1, 0, rows - 1, 1).format.numberFormat = "0.####";
  sheet.getRangeByIndexes(1, 1, rows - 1, cols - 1).format.numberFormat = "0.0000";
  sheet.getRangeByIndexes(0, 0, rows, cols).format.autofitColumns();
  sheet.getRange("A:A").format.columnWidth = 24;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
  sheet.showGridLines = false;
}

async function exportWorkbook(name, payload, sheets) {
  const workbook = await importTemplate(name);
  for (const spec of sheets) {
    const sheet = workbook.worksheets.getItem(spec.sheetName);
    const old = sheet.getUsedRange();
    if (old) old.clear({ applyTo: "all" });
    writeMatrix(sheet, payload.time, payload.distance, payload[spec.key]);
  }
  workbook.recalculate();
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 50 },
    summary: `${name} final formula error scan`,
  });
  if (errors.ndjson && errors.ndjson.includes("match")) {
    console.log(errors.ndjson);
  }
  const file = await SpreadsheetFile.exportXlsx(workbook);
  await file.save(path.join(outputDir, name));
}

await fs.mkdir(outputDir, { recursive: true });
const result1 = await loadJson("result1.json");
const result2 = await loadJson("result2.json");
const result3 = await loadJson("result3.json");
const result4 = await loadJson("result4.json");

// 所有输入先验证，避免到第四份才发现坏数据而留下部分更新的输出。
validatePayload(result1, ["temperature", "moisture"], "result1");
validatePayload(result2, ["temperature", "moisture"], "result2");
validatePayload(result3, ["moisture"], "result3");
validatePayload(result4, ["moisture"], "result4");
if (process.argv.includes("--check-inputs")) {
  console.log("四份导出输入检查通过；未修改 Excel。");
  process.exit(0);
}

await exportWorkbook("result1.xlsx", result1, [
  { sheetName: "温度", key: "temperature" },
  { sheetName: "水分浓度", key: "moisture" },
]);
await exportWorkbook("result2.xlsx", result2, [
  { sheetName: "温度", key: "temperature" },
  { sheetName: "水分浓度", key: "moisture" },
]);
await exportWorkbook("result3.xlsx", result3, [
  { sheetName: "Sheet1", key: "moisture" },
]);
await exportWorkbook("result4.xlsx", result4, [
  { sheetName: "Sheet1", key: "moisture" },
]);

console.log(`Exported result workbooks to ${outputDir}`);
