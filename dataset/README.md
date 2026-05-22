# Data Sources

This directory contains real cloud workload data used to calibrate simulation parameters for the DQN-Cloud scheduler.

---

## 1. Google Cluster Trace 2019

**Source:** Google Borg Cluster Management System  
**Documentation:** https://github.com/google/cluster-data/blob/master/ClusterData2019.md  

### How to Download

The trace is available in JSON format from Google Cloud Storage. Each cell's data is stored in a bucket named `clusterdata_2019_${CELL}` where `${CELL}` is one of: `a`, `b`, `c`, `d`, `e`, `f`, `g`, `h`.

1. Create a `data` directory inside `google_cluster` and navigate to it:
```bash
mkdir google_cluster/data
cd google_cluster/data
```

2. Download a shard using `gsutil`:
```bash
gsutil cp gs://clusterdata_2019_a/instance_events-000000000000.json.gz data/google_cluster/
gsutil cp gs://clusterdata_2019_a/machine_events-000000000000.json.gz data/google_cluster/
```

2. Decompress using GZIP:
```bash
gunzip data/google_cluster/instance_events-000000000000.json.gz
gunzip data/google_cluster/machine_events-000000000000.json.gz
```



> **Note:** All resources are normalized to [0,1]. Scale to simulation units by multiplying by VM capacity.

---

## 2. Alibaba Cluster Trace 2018

**Source:** Alibaba Production Cluster  
**Documentation:** https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018  

### How to Download

A download script `fetchData.sh` is included in `dataset/alibaba/`. To download all files:
 
```bash
cd dataset/alibaba
bash fetchData.sh
```
 
To download `batch_task` only (most relevant file):
 
```bash
curl -C - -o dataset/alibaba/data/batch_task.tar.gz \
  'http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/batch_task.tar.gz'
tar -xzf dataset/alibaba/data/batch_task.tar.gz -C dataset/alibaba/data/
```


## 3. GoCJ — Google Cloud Jobs Dataset

**Source:** Derived from Google Cluster Traces via Monte Carlo simulation  
**DOI:** https://doi.org/10.3390/data3040038  
**Download:** https://www.researchgate.net/publication/327950230  

### How to download

1. Go to: https://data.mendeley.com/datasets/b7bp6xhrcd/1
2. Click **Download All** 
3. Extract and place files in `data/gocj/`




