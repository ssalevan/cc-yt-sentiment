#!/usr/bin/python
# Analyzes sentiment from YouTube comments using an NLTK-based sentiment
# classifier.  Output will consist of a sorted file in the following
# format:
#
# <sentiment score between -1 and 1>  "<comma-separated YouTube URLs>"
#
# Copyright 2013 Steve Salevan (steve.salevan@gmail.com)
# 
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#


import gzip
import json
import operator
import os
import pickle
import StringIO


import BeautifulSoup
import boto
import mrjob
import nltk
import yaml


import nltk.tokenize as tokenize
from BeautifulSoup import BeautifulSoup
from boto.s3.connection import S3Connection
from gzip import GzipFile
from mrjob.job import MRJob
from StringIO import StringIO


CC_BUCKET = "aws-publicdatasets"


def UnpickleObject(file_loc):
  """
  Returns an unpickled Python object contained within the file located at
  file_loc.
  """

  pickle_file = open(file_loc, "rb")
  unpickled_obj = pickle.load(pickle_file)
  pickle_file.close()
  return unpickled_obj


def GetArcFile(s3, bucket, info):
  """
  Retrieves the GzipFile corresponding to a page in the Common Crawl dataset
  as described by the supplied info dictionary.
  """

  bucket = s3.lookup(bucket)
  keyname = "/common-crawl/parse-output/segment/{arcSourceSegmentId}/" \
            "{arcFileDate}_{arcFileParition}.arc.gz".format(**info)
  key = bucket.lookup(keyname)
  start = info["arcFileOffset"]
  end = start + info["compressedSize"] - 1
  headers = {"Range" : "bytes=%s-%s" % (start, end)}
  chunk = StringIO(
    key.get_contents_as_string(headers = headers)
  )
  return GzipFile(fileobj = chunk)


def ReadYamlFile(file_loc):
  """
  Deserializes YAML data contained within the file located at the supplied
  file location.
  """

  yaml_file = open(file_loc, "r")
  data_map = yaml.safe_load(yaml_file)
  yaml_file.close()
  return data_map


def BagOfWords(words):
  """
  Returns a 'bag of words' dictionary, mapping words to True values, which will
  be used by NLTK to classify the sentiment of the provided words.
  """

  return dict([word, True] for word in words)


class YouTubeSentimentAnalysis(MRJob):

  def mapper_init(self):
    # Unpickles the trained NLTK sentiment classifier object.
    self.classifier = UnpickleObject("classifier.pkl")
    mrjob_conf = ReadYamlFile(".mrjob.conf")
    # Instantiates a connection to Amazon S3 with the credentials configured
    # for the current MapReduce job.
    self.s3 = S3Connection(mrjob_conf["runners"]["emr"]["aws_access_key_id"],
        mrjob_conf["runners"]["emr"]["aws_secret_access_key"])

  def mapper(self, _, line):
    # Parses JSON record mapping crawled page to location in CC S3 bucket.
    cur_page_info = json.loads(line)
    # Retrieves the GzipFile corresponding to the crawled page from S3.
    page_arcfile = GetArcFile(self.s3, CC_BUCKET, cur_page_info)
    # Parses the page contents via BeautifulSoup.
    doc_soup = BeautifulSoup(page_arcfile)
    self.increment_counter('YouTube', 'num_videos', 1)
    # Locates all the comments on a page by finding all tags that match:
    # <div class='comment-text'>
    sentiment_scores = []
    for comment in doc_soup.findAll("div", {"class": "comment-text"}):
      # Classifies the positive and negative sentiment probabilities from the
      # textual content of the current YouTube comment.
      visible_text = unicode(''.join(comment.findAll(text = True)))
      word_features = BagOfWords(tokenize.word_tokenize(visible_text))
      prob_dist = self.classifier.prob_classify(word_features)
      # Finds the difference between negative and positive probabilities.
      probabilities = [prob_dist.prob('neg'), prob_dist.prob('pos')]
      sentiment_score = max(probabilities) - min(probabilities)
      # Negates the 'sentiment score' if the comment has a negative sentiment.
      if 'neg' in prob_dist.generate():
        sentiment_score *= -1
      sentiment_scores.append(sentiment_score)
      self.increment_counter('YouTube', 'num_comments', 1)
    # If at least one sentiment score exists, derives the average sentiment
    # score; otherwise returns a bogon value of 2.0.
    avg_sentiment_score = 2.0
    if len(sentiment_scores) > 0:
      avg_sentiment_score = reduce(operator.add, sentiment_scores) / \
          float(len(sentiment_scores))
    yield avg_sentiment_score, cur_page_info['url']

  def reducer(self, score, urls):
    yield score, ",".join(urls)


if __name__ == '__main__':
  YouTubeSentimentAnalysis.run()
