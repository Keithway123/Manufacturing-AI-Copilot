---
doc_id: mes_work_order_status
title: MES工单状态说明
department: 制造运营部
doc_type: manual
version: v1.0
access_level: internal
---

# MES工单状态说明

## 适用范围

本文档用于说明 MES 系统中生产工单的常见状态、状态含义和操作边界，适用于生产、计划、质量和仓库人员。

## 常见工单状态

### Created

工单已创建，但尚未下达到产线。此状态下只能查看工单信息，不能进行生产报工。

### Released

工单已下达，产线可以开始生产准备。生产人员需要确认物料、设备、工装和作业指导书是否齐套。

### In Progress

工单正在生产中。系统会记录开工时间、良品数、不良品数、报工人员和关键过程数据。

### Paused

工单暂停。常见原因包括设备故障、物料短缺、质量异常、计划调整或人员不足。

### Completed

工单已完成。产量、质量数据和完工时间已提交，后续修改需要主管审批。

### Closed

工单已关闭。关闭后不能再进行报工、返工录入或产量修改。

## 状态流转规则

- Created 状态只能流转为 Released 或 Cancelled。
- Released 状态可以流转为 In Progress 或 Cancelled。
- In Progress 状态可以流转为 Paused 或 Completed。
- Paused 状态恢复生产后应回到 In Progress。
- Completed 状态经审核后流转为 Closed。

## 操作注意事项

- 工单未 Released 前，不允许产线报工。
- Paused 状态必须填写暂停原因。
- Completed 后如需修改数据，应提交主管审批。
- 质量异常未关闭前，不建议将工单转为 Closed。

## 常见问题

### 工单无法开工

优先检查工单是否为 Released 状态，再检查物料齐套状态和设备状态。

### 工单无法完工

检查是否存在未处理的不良品记录、未完成的质量检验或未提交的报工数据。
