# Pytheas: Pattern-based Table Discovery in CSV Files
Code repository for [Pytheas: Pattern-based Table Discovery in CSV Files](http://www.vldb.org/pvldb/vol13/p2075-christodoulakis.pdf), presented at VLDB 2020.
[![PytheasVLDB2020](http://img.youtube.com/vi/PHc-tGeZeD0/0.jpg)](https://www.youtube.com/watch?v=PHc-tGeZeD0&ab_channel=VLDB2020 "Video Title")


#### Abstract

ABSTRACT
CSV is a popular Open Data format widely used in a variety of domains for its simplicity and effectiveness in storing and disseminating data. Unfortunately, data published in this format often does not conform to strict specifications, making automated data extraction from CSV files a painful task. While table discovery from HTML pages or spreadsheets has been studied extensively, extracting tables from CSV files still poses a considerable challenge due to their loosely defined format and limited embedded metadata.  
In this work we lay out the challenges of discovering tables in CSV files, and propose Pytheas: a principled method for automatically classifying lines in a CSV file and discovering tables within it based on the intuition that tables maintain a coherency of values in each column. We evaluate our methods over two manually annotated data sets: 2000 CSV files sampled from four Canadian Open Data portals, and 2500 additional files sampled from Canadian, US, UK and Australian portals. Our comparison to state-of-the-art approaches shows that Pytheas is able to successfully discover tables with precision and recall of over 95.9% and 95.7% respectively, while current approaches achieve around 89.6% precision and 81.3% recall. Furthermore, Pytheas’s accuracy for correctly classifying all lines per CSV file is 95.6%, versus a maximum of 86.9% for compared approaches. Pytheas generalizes well to new data, with a table discovery Fmeasure above 95% even when trained on Canadian data and applied to data from different countries. Finally, we introduce a confidence measure for table discovery and demonstrate its value for accurately identifying potential errors. 

#### How to cite

Christina Christodoulakis, Eric B. Munson, Moshe Gabel, Angela Demke Brown, and Renée J. Miller. Pytheas: Pattern-based Table Discovery in CSV Files. PVLDB, 13(11): 2075-2089, 2020. DOI: https://doi.org/10.14778/3407790.3407810

```bib
@article{DBLP:journals/pvldb/Christodoulakis20,
  author    = {Christina Christodoulakis and
               Eric Munson and
               Moshe Gabel and
               Angela Demke Brown and
               Ren{\'{e}}e J. Miller},
  title     = {Pytheas: Pattern-based Table Discovery in {CSV} Files},
  journal   = {Proc. {VLDB} Endow.},
  volume    = {13},
  number    = {11},
  pages     = {2075--2089},
  year      = {2020},
  url       = {http://www.vldb.org/pvldb/vol13/p2075-christodoulakis.pdf},
  timestamp = {Mon, 17 Aug 2020 18:32:39 +0200},
  biburl    = {https://dblp.org/rec/journals/pvldb/Christodoulakis20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


*Updated code and detailed instructions for use will be added in the coming weeks. 
