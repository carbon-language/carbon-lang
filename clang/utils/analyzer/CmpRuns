#!/usr/bin/env python

"""
CmpRuns - A simple tool for comparing two static analyzer runs to determine
which reports have been added, removed, or changed.

This is designed to support automated testing using the static analyzer, from
two perspectives: 
  1. To monitor changes in the static analyzer's reports on real code bases, for
     regression testing.

  2. For use by end users who want to integrate regular static analyzer testing
     into a buildbot like environment.
"""

import os
import plistlib

#

class multidict:
    def __init__(self, elts=()):
        self.data = {}
        for key,value in elts:
            self[key] = value
    
    def __getitem__(self, item):
        return self.data[item]
    def __setitem__(self, key, value):
        if key in self.data:
            self.data[key].append(value)
        else:
            self.data[key] = [value]
    def items(self):
        return self.data.items()
    def values(self):
        return self.data.values()
    def keys(self):
        return self.data.keys()
    def __len__(self):
        return len(self.data)
    def get(self, key, default=None):
        return self.data.get(key, default)
    
#

class AnalysisReport:
    def __init__(self, run, files):
        self.run = run
        self.files = files

class AnalysisDiagnostic:
    def __init__(self, data, report, htmlReport):
        self.data = data
        self.report = report
        self.htmlReport = htmlReport

    def getReadableName(self):
        loc = self.data['location']
        filename = self.report.run.getSourceName(self.report.files[loc['file']])
        line = loc['line']
        column = loc['col']

        # FIXME: Get a report number based on this key, to 'distinguish'
        # reports, or something.
        
        return '%s:%d:%d' % (filename, line, column)

    def getReportData(self):
        if self.htmlReport is None:
            return "This diagnostic does not have any report data."

        return open(os.path.join(self.report.run.path,
                                 self.htmlReport), "rb").read() 

class AnalysisRun:
    def __init__(self, path, opts):
        self.path = path
        self.reports = []
        self.diagnostics = []
        self.opts = opts

    def getSourceName(self, path):
        if path.startswith(self.opts.root):
            return path[len(self.opts.root):]
        return path

def loadResults(path, opts):
    run = AnalysisRun(path, opts)

    for f in os.listdir(path):
        if (not f.startswith('report') or
            not f.endswith('plist')):
            continue

        p = os.path.join(path, f)
        data = plistlib.readPlist(p)

        # Ignore empty reports.
        if not data['files']:
            continue

        # Extract the HTML reports, if they exists.
        if 'HTMLDiagnostics_files' in data['diagnostics'][0]:
            htmlFiles = []
            for d in data['diagnostics']:
                # FIXME: Why is this named files, when does it have multiple
                # files?
                assert len(d['HTMLDiagnostics_files']) == 1
                htmlFiles.append(d.pop('HTMLDiagnostics_files')[0])
        else:
            htmlFiles = [None] * len(data['diagnostics'])
            
        report = AnalysisReport(run, data.pop('files'))
        diagnostics = [AnalysisDiagnostic(d, report, h) 
                       for d,h in zip(data.pop('diagnostics'),
                                      htmlFiles)]

        assert not data

        run.reports.append(report)
        run.diagnostics.extend(diagnostics)

    return run

def compareResults(A, B):
    """
    compareResults - Generate a relation from diagnostics in run A to
    diagnostics in run B.

    The result is the relation as a list of triples (a, b, confidence) where
    each element {a,b} is None or an element from the respective run, and
    confidence is a measure of the match quality (where 0 indicates equality,
    and None is used if either element is None).
    """

    res = []

    # Quickly eliminate equal elements.
    neqA = []
    neqB = []
    eltsA = list(A.diagnostics)
    eltsB = list(B.diagnostics)
    eltsA.sort(key = lambda d: d.data)
    eltsB.sort(key = lambda d: d.data)
    while eltsA and eltsB:
        a = eltsA.pop()
        b = eltsB.pop()
        if a.data == b.data:
            res.append((a, b, 0))
        elif a.data > b.data:
            neqA.append(a)
            eltsB.append(b)
        else:
            neqB.append(b)
            eltsA.append(a)
    neqA.extend(eltsA)
    neqB.extend(eltsB)

    # FIXME: Add fuzzy matching. One simple and possible effective idea would be
    # to bin the diagnostics, print them in a normalized form (based solely on
    # the structure of the diagnostic), compute the diff, then use that as the
    # basis for matching. This has the nice property that we don't depend in any
    # way on the diagnostic format.

    for a in neqA:
        res.append((a, None, None))
    for b in neqB:
        res.append((None, b, None))

    return res

def main():
    from optparse import OptionParser
    parser = OptionParser("usage: %prog [options] [dir A] [dir B]")
    parser.add_option("", "--root", dest="root",
                      help="Prefix to ignore on source files",
                      action="store", type=str, default="")
    parser.add_option("", "--verbose-log", dest="verboseLog",
                      help="Write additional information to LOG [default=None]",
                      action="store", type=str, default=None,
                      metavar="LOG")
    (opts, args) = parser.parse_args()

    if len(args) != 2:
        parser.error("invalid number of arguments")

    dirA,dirB = args

    # Load the run results.
    resultsA = loadResults(dirA, opts)
    resultsB = loadResults(dirB, opts)
    
    # Open the verbose log, if given.
    if opts.verboseLog:
        auxLog = open(opts.verboseLog, "wb")
    else:
        auxLog = None

    diff = compareResults(resultsA, resultsB)
    for res in diff:
        a,b,confidence = res
        if a is None:
            print "ADDED: %r" % b.getReadableName()
            if auxLog:
                print >>auxLog, ("('ADDED', %r, %r)" % (b.getReadableName(),
                                                        b.getReportData()))
        elif b is None:
            print "REMOVED: %r" % a.getReadableName()
            if auxLog:
                print >>auxLog, ("('REMOVED', %r, %r)" % (a.getReadableName(),
                                                          a.getReportData()))
        elif confidence:
            print "CHANGED: %r to %r" % (a.getReadableName(),
                                         b.getReadableName())
            if auxLog:
                print >>auxLog, ("('CHANGED', %r, %r, %r, %r)" 
                                 % (a.getReadableName(),
                                    b.getReadableName(),
                                    a.getReportData(),
                                    b.getReportData()))
        else:
            pass

    print "TOTAL REPORTS: %r" % len(resultsB.diagnostics)
    if auxLog:
        print >>auxLog, "('TOTAL', %r)" % len(resultsB.diagnostics)

if __name__ == '__main__':
    main()
