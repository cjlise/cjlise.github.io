---
title: "Setting up a Hadoop cluster on Windows using Docker and WSL2"
date: 2020-09-26
categories: hadoop-spark
tags: [Big Data, Distributed Processing, Linux, Hadoop, Docker]
header: 
   image: "/images/HadoopSpark/Hadoop-Spark-Logo.png"
excerpt: "Big Data, Distributed Processing, Linux, Hadoop, Docker"
---

  

I wanted to setup a Hadoop cluster as a playground on my Windows 10 laptop. I thought that using Docker with the new WSL2 (Windows Sub-system Linux version 2) included in Windows 10 version 0420 could be a solution. Indeed Docker can use WSL2 to run natively Linux on Windows. I basically followed the tutorial [How to set up a Hadoop cluster in Docker](https://clubhouse.io/developer-how-to/how-to-set-up-a-hadoop-cluster-in-docker/) that is normally designed for a Linux host machine running docker (and not Windows). 




## 1. Install Docker on Windows 
I'm currently using docker desktop version 2.3.0.3 from the stable channel. But any version that supports WSL2 should work.
The corresponding engine version is 19.03.8 and docker-compose version is 1.25.5:

![Docker version](/images/HadoopSpark/SetupHadoop-01-Versions.png "Docker version")

You can confirm that docker is running properly by launching a web server: 

```bash
docker run -d -p 80:80 --name myserver nginx
```


## 2. Setting up Hadoop cluster using Docker
Use git to download the the Hadoop Docker files from the [Big Data Europe repository](https://github.com/big-data-europe/docker-hadoop):

```bash
git clone git@github.com:big-data-europe/docker-hadoop.git
```
Deploy the docker cluster using the command: 

 ```bash
docker-compose up -d
```
You can check that the containers are running using: 

 ```bash
docker ps
```
You can also double check with the Docker dashboard: 

![Docker Dashboard](/images/HadoopSpark/SetupHadoop-02-docker-dashboard.png "Docker Dashboard")


And the current status can also be checked using the web page [http://localhost:9870](http://localhost:9870):

![Hadoop Overview](/images/HadoopSpark/SetupHadoop-03-Overview.png "Hadoop Overview")


## 3. Testing the Hadoop cluster

We will test the Hadoop cluster running the Word Count example. 

* Open a terminal session on the namenode
```bash
docker exec -it namenode bash
```
This will open a session on the namenode for the root user. 

* Create some simple text files to be used by the wordcount program 
```bash
cd /tmp
mkdir input
echo "Hello World" >input/f1.txt
echo "Hello Docker" >input/f2.txt
```

* Create a hdfs directory named inut
```bash
hadoop fs -mkdir -p input
```
* Put the input files in all the datanodes on HDFS
```bash
hdfs dfs -put ./input/* input
```
* Download on the host pc (e.g in the directory on top of the hadoop cluster directory) the word count program from [this link](https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-mapreduce-examples/2.7.1/hadoop-mapreduce-examples-2.7.1-sources.jar)

* Run the command below in a terminal on the Windows host to identify the namenode container id:
```bash
docker container ls
```
![namenode id](/images/HadoopSpark/SetupHadoop-04-NameNodeID.png "namenode id")

* Use the command below on the Windows host to copy the word count program in the namenode container: 

```bash
docker cp ../hadoop-mapreduce-examples-2.7.1-sources.jar afb235f8629c:/tmp
```
* Run the word count program in the namenode: 
```bash
hadoop jar hadoop-mapreduce-examples-2.7.1-sources.jar org.apache.hadoop.examples.WordCount input output
```
The program should display something like: 

![Hadoop Job](/images/HadoopSpark/SetupHadoop-05-Job.png "Hadoop Job")

* Print the output of the word count program 
```bash
hdfs dfs -cat output/part-r-00000
```
![Hadoop Output](/images/HadoopSpark/SetupHadoop-06-output.png "Hadoop Output")

That's all ! 




	