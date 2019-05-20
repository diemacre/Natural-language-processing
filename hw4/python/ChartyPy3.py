#!/usr/bin/env python3


"""
@author: Diego Martin Crespo
@id: A20432558
    
ChartyPy3.py

This is a small incremental bottom-up chart parser for context free grammars.

(C) 2005-2011 by Damir Cavar <damir@cavar.me>

This code is written and distributed under the
Lesser GNU General Public License version 3 or newer.

See http://www.gnu.org/copyleft/lesser.html for details
on the license or the the file lgpl-3.0.txt that should always be
distributed with this code.


Used data structures:

chart:
   list of edges

edge:
   list of integers and symbols
   [start, end, dotindex, LHS, RHS]
   start:    integer, start of the edge
   end:      integer, end of the edge
   dotindex: integer, position of the dot in the RHS
   LHS:      string, left-hand side symbol
   RHS:      list of strings, symbols in right-hand side


Properties:
Incremental (left-to-right) bottom-up chart parser.
Select only potentially appropriate rules from grammar
   - length of RHS is less or equal to remaining words/symbols


Processing steps:
   Word by word:
      Initialise chart with word (add edge for word)
      Do until no further improvement:
         Add new rules from grammar that consume inactive edges
         Apply the fundamental rule to induce new edges


Calling via command line:
If ChartyPy3.py is made executable, one can call it:

./ChartyPy3.py -g PSG1.txt -i "John loves Mary"

or start Python with the script otherwise:

python ChartyPy3.py -g PSG1.txt -i "John loves Mary"

Start the script with as:

python ChartyPy3.py -h

for instructions about the parameters.


This code can be opimized. However, its main purpose is to help students understand a simple algorithm for chart parsing. If there are any bugs, please let me know:

Damir Cavar <damir@cavar.me>
"""

__author__  = "Damir Cavar <damir@cavar.me>"
__date__    = "$May 29, 2005 10:36:30 AM$"
__version__ = "0.5"


import sys, PSGParse3
import argparse
import math

DEBUG = False       # set this to 0 if you do not want tracking
QTREE = False


def isActive(edge):
   """Return 1 if edge is active, else return 0."""
   if edge[2] < len(edge[4]): return True
   return False


def isInactive(edge):
   """Return True if edge is active, else returns False."""
   if edge[2] >= len(edge[4]): return True
   return False


def match(aedge, iedge):
   """Returns True if the active edge and the inactive edge match,
      otherwise False.
   """
   if aedge[1] == iedge[0]:
      if aedge[4][aedge[2]] == iedge[3]: return True
   return False


def getParse(inputlength, chart, grammar):
   '''TODO:
      MODIFICATION:
      This method has been modified so it returns two arrays
      - one contains the possible trees of the sentence 
      - the other contains the corresponding probability to those possible trees
         if the prob gotten is 0 it prints that the tree has not prob defined for its rules
   '''
   parses = [] 
   probs = [] #auxilary variable which will contain the probs for each tree
   for i in range(len(chart)):
      if not isActive(chart[i]):
         if chart[i][0] == 0 and chart[i][1] == inputlength: # got spanning edge
            print("Successfully parsed!")
            prob = 0
            stru, prob = struct2Str(chart[i], chart, grammar, prob) #call the funtion that will create the tree and calculate its probability
            if len(stru)>0:
               stru = stru[1:-1]
            parses.append(stru)
            if prob!=0:
               prob = math.exp(prob)
            else:
               prob = 'File with no probability defined or defined to "0" by default'
            probs.append(prob)
   
   return parses, probs



def struct2Str(edge, chart, grammar, prob=0):
   """TODO:
      MODIFICATION: this method has being modified so it returns a string representation of the parse with
      labled brackets (tree) and its probability

      Parameters:
      edges - the lsit of edges that make a parse
      chart - the current chart (list of edges)
      grammar - grammar of the file
      prob - probability accumulated in log units
   """
   tmpstr = ""
   edgenums = edge[5] #take the next edges of the chart
   tmpstr = "".join((tmpstr, "[", grammar.id2s(edge[3]))) #add the grammar of that edge to the tree
   
   aux = []
   for x in edge[4]:
      aux.append(grammar.id2s(x))
   key = (grammar.id2s(edge[3]), tuple(aux)) # the correct format of the dictionary key is created to then search its probability
   #the probability is calculated for each rule
   if 0.0 != float(grammar.getProb(key)):
      prob += math.log(float(grammar.getProb(key))) #call an auxilary funtion created in PSGParse3.py that will return the probability of the rule pased as key of a dictionary
   for x in edge[4]:
      if grammar.isTerminal(x): # if is terminal 
         tmpstr = " [".join((tmpstr, grammar.id2s(x))).join(" ]") #the grammar is added to the tree
      else:
         struc, prob = struct2Str(chart[edgenums[0]], chart, grammar, prob) # call recursevely with the next edge from the chart and new acumulated probability
         tmpstr = " ".join((tmpstr, struc)) #add the grammar to the tree
         edgenums = edgenums[1:] #update the edges removing the previous visited
      
   tmpstr = "".join((tmpstr, " ]")) #add grammar to the tree
   return tmpstr, prob# return the tree and probability accumulated


def edgeStr(edge, grammar):
   """ """
   return str( (edge[0], edge[1], edge[2],
           grammar.id2s(edge[3]),
           grammar.idl2s(edge[4]),
           edge[5]) )


def ruleInvocation(lststart, chart, inputlength, grammar):
   """Add all the rules of the grammar to the chart that
      are relevant:
      Find the rule with the LHS of edge as the leftmost RHS
      symbol and maximally the remaining length of the input.

      Parameters:
      lststart - start position at edge in chart
      chart - the current chart
      inputlength - the length of the input sentence
      grammar - the grammar object raturned by PSGParse3
   """
   change = False
   for i in range(lststart, len(chart)):
      if chart[i][2] >= len(chart[i][4]): # only inactive edge
         (start, end, index, lhs, rhs, consumed) = chart[i]
         for k in grammar.rhshash.get(lhs, ()):
            if len(k[1]) > inputlength - start:
               continue
            newedge = ( start, end, 1, k[0], k[1], (i,) )
            if newedge in chart:
               continue
            chart.append(newedge)
            change = True
            if DEBUG:
               print("RI Adding edge:", edgeStr(newedge, grammar))
   return change


def fundamentalRule(chart, grammar):
   """The fundamental rule of chart parsing generates new edges by
      combining fitting active and inactive edges.

      Parameters:
      chart - the current chart
   """
   change = False
   for aedge in chart:
      if isActive(aedge):
         for k in range(len(chart)):
            if isInactive(chart[k]):
               if match(aedge, chart[k]):
                  newedge = (aedge[0], chart[k][1], aedge[2] + 1,
                             aedge[3], aedge[4], tuple(list(aedge[5]) + [ k ]))
                  if newedge not in chart:
                     chart.append(newedge)
                     change = True
                     if DEBUG:
                        print("FR Adding edge:", edgeStr(newedge, grammar))
   return change


def parse(inp, grammar):
   """Parse a list of tokens.

      Arguments:
      inp = a list of tokens
      grammar = an object returned by PSGParse3
   """
   chart = []
   inputlength = len(inp)

   chartpos = 0  # remember start-position in chart
   for i in range(inputlength):
      # initialize with input token
      rules = grammar.rhshash.get(grammar.symb2id[inp[i]], ( ("", ()) ) )
      for rule in rules:
         if rule[0]:
            chart.append( ( i, i + 1, 1, rule[0], rule[1], () ) )
      if DEBUG:
         print("Adding edge:", edgeStr(chart[len(chart) - 1], grammar))
      change = 1
      while change:
         change = 0
         chartlen = len(chart)
         if ruleInvocation(chartpos, chart, inputlength, grammar):
            change = 1
         chartpos = chartlen  # set pointer to new edge in chart
         if fundamentalRule(chart, grammar):
            change = 1
   if DEBUG:
      print("Chart:")
      for i in range(len(chart)):
         if isActive(chart[i]):
            print(i, "Active:", end=" ")
         else:
            print(i, "Inactive:", end=" ")
         print(edgeStr(chart[i], grammar))
   if QTREE:
      return getQtreeParse(inputlength, chart, grammar)
   return getParse(inputlength, chart, grammar)


def printParses(parses, probs):
   """TODO: 
      MODIFICATION: Prints the parse as brackated string to the screen and their probability
      Arguments:
      parses - array of tree
      probs  - array of tree probabilities
      """
   for i in range(len(parses)):
      print("Tree:", parses[i])
      print("Prob:", probs[i])

if __name__ == "__main__":
   usage = "usage: %(prog)s [options]"
   parser = argparse.ArgumentParser(prog="ChartyPy", usage=usage,
            description='A chart parser, based on the Earley algorithm.',
            epilog="(C) 2005-2011 by Damir Cavar <damir@cavar.me>")
   parser.add_argument('--version', action='version', version="ChartyPy "+__version__)
   parser.add_argument("-g", "--grammar", dest="grammar", required=True,
            help="name of the file with the context-free grammar")
   parser.add_argument("-i", "--input", dest="sentence", required=True,
            help="input sentence, e.g. \"John kissed Mary\"")
   parser.add_argument("-l", "--latex", dest="latex", action="store_true",
            required=False,
            help="output of parse structure in LaTeX notation for qtree.sty")
   parser.add_argument("-q", "--quiet",
            action="store_false", dest="DEBUG", default=True,
            help="don't print the chart content  [default True]")
   args = parser.parse_args()
   if args:
      DEBUG = args.DEBUG
      QTREE = args.latex
      try:
         mygrammar = PSGParse3.PSG(args.grammar) # initialization of the grammar
      except IOError:
         print("Cannot load grammar:, args.grammar")
      else:
         parses, probs = parse(args.sentence.split(), mygrammar)
         printParses(parses, probs)

