---
title: "Hadoop Map Reduce Example Using Yarn"
date: 2020-08-30
categories: hadoop-spark
tags: [Big Data, Distributed Processing, Yarn, Python, Linux, Cloudera]
header: 
   image: "/images/HadoopSpark/Hadoop-Spark-Logo.png"
excerpt: "Big Data, Distributed Processing, Yarn, Python, Linux, Cloudera"
---

  
# Hadoop Map Reduce Example Using Yarn  
The purpose of this example is to show how to carry out a Map Reduce task to find the top used word in the [Beowulf Ebook](https://www.gutenberg.org/cache/epub/16328/pg16328.txt) using Yarn with 2 Python scripts. 

## Python scripts
### map.py
This script is used to split the input stream in words, then print on the output stream the tuples (word, 1): 

```python
#!/usr/bin/env python
"""map.py"""

import sys

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    words = line.split()
    # increase counters
    for word in words:
        # write the results to STDOUT (standard output);
        # what we output here will be the input for the
        # Reduce step, i.e. the input for reducer.py
        #
        # tab-delimited; the trivial word count is 1
        print '%s\t%s' % (word, 1)

```

### reduce.py
The script take as input the output of the map.py script and increase the word count when a duplicated word is identified. 
At the end reduce prints each word and its word count:

```python

#!/usr/bin/env python
"""reduce.py"""

from operator import itemgetter
import sys

current_word = None
current_count = 0
word = None

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    # parse the input we got from mapper.py
    word, count = line.split('\t', 1)

    # convert count (currently a string) to int
    try:
        count = int(count)
    except ValueError:
        # count was not a number, so silently
        # ignore/discard this line
        continue

    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    if current_word == word:
        current_count += count
    else:
        if current_word:
            # write result to STDOUT
            print '%s\t%s' % (current_word, current_count)
        current_count = count
        current_word = word

# do not forget to output the last word if needed!
if current_word == word:
    print '%s\t%s' % (current_word, current_count)

``` 
 


## Map Reduce with Yarn
The Yarn command to launch the map reduce process is listed below: 
```bash
yarn jar /usr/hdp/current/hadoop-mapreduce-client/hadoop-streaming.jar \
	-file /home/jose.lise-dsti/map.py \
	-mapper /home/jose.lise-dsti/map.py \
	-file /home/jose.lise-dsti/reduce.py \
	-reducer /home/jose.lise-dsti/reduce.py \
	-input /user/jose.lise-dsti/raw/pg16328.txt \
	-output /user/jose.lise-dsti/python-output
```

Here is the output that we get after execution: 
![yarn command output](/images/HadoopSpark/YarnMapReduce.jpg "yarn command output")


And the results: 
![yarn results](/images/HadoopSpark/YarnMapReduce.jpg "yarn results")





	