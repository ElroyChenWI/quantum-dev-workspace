# IBM Quantum 雲端整合 Audit

> 本文件記錄 Quantum Dev Workspace 與 IBM Quantum 雲端的實際整合狀態。
> 用途：作為「本機 → 雲端」打通之證明，並作為更新 README / repo 的依據。

---

## 1. 基本資訊

| 項目 | 內容 |
|------|------|
| 日期 | 2026-08-12 |
| 平台 | IBM Quantum Platform |
| 方案 | Open Plan（免費，trial 帳號，每月 10 分鐘量子時間）|
| API | qiskit-ibm-runtime `0.49.0`，channel = `ibm_quantum_platform` |
| 連線方式 | Token 存於本機 `.env`（已被 .gitignore 排除，不提交）|

## 2. 可用 Backend（真實量子處理器）

| Backend | Qubits | 狀態 |
|---------|--------|------|
| ibm_fez | 156 | operational |
| ibm_kingston | 156 | operational |
| ibm_marrakesh | 156 | operational |

> Open Plan 下無雲端模擬器，直接使用真實 156-qubit 處理器。

## 3. 提交的任務

| 項目 | 內容 |
|------|------|
| Job ID | `d9u09ql35hes73fj9rgg` |
| Backend | ibm_kingston（156 qubits）|
| 電路 | H2 ansatz（2 qubits，4 參數：`[0.1, 0.1, 0.05, 0.05]`）|
| Observable | H2 哈密頓量（5 項 Pauli，見 `h2_hamiltonian.py`）|
| 本機參考值 | **-1.047914 Ha**（StatevectorEstimator，理想無雜訊）|
| 精確基態能量 | -1.857275 Ha（對角化）|
| 提交時間 | 2026-08-12 ~13:30（UTC+8）|
| 狀態 | 完成（已回傳）|

## 4. 雲端結果

| 項目 | 內容 |
|------|------|
| IBM 雲端能量（真硬體）| **-1.088185 Ha** |
| 本機理想值（無雜訊）| -1.047914 Ha |
| 差異（vs 本機理想 expectation）| 4.03e-02 Ha |
| 解讀 | 這個差異主要反映真實量子硬體的 noise、shot statistics、transpilation/layout 與未做 error mitigation 的實機效應；本機 statevector 是無雜訊理想參考值。 |

## 5. 技術重點（從錯誤中學到的）

- IBM Runtime 自 2024-03 起**要求電路先轉譯成目標硬體的 ISA** 才能提交
  → 需用 `generate_preset_pass_manager(backend=...)` 轉譯，並以
    `observable.apply_layout(...)` 對應實體 qubit
- channel 名稱在 0.49 版為 `ibm_quantum_platform`（非舊版 `ibm_quantum`）

## 6. 安全備註

- Token 不應寫入任何會被提交的檔案；僅存於 `.env`（gitignored）
- 本輪 token 曾出現於對話紀錄，**建議 demo 結束後於 IBM 重新產生一把新的**

## 7. 更新完成事項

- [x] 將雲端結果填入第 4 節
- [x] README「VQE 跨框架專案」加入 IBM 雲端實測結果
- [x] README 框架表：Qiskit 後端更新為「本機模擬 + IBM Quantum 雲端實測」
- [x] 新增本文件到 repo（`docs/IBM_CLOUD_AUDIT.md`）
