#!/usr/bin/env python

"""
Static Analyzer qualification infrastructure.

The goal is to test the analyzer against different projects, check for failures,
compare results, and measure performance.

Repository Directory will contain sources of the projects as well as the 
information on how to build them and the expected output. 
Repository Directory structure:
   - ProjectMap file
   - Historical Performance Data
   - Project Dir1
     - ReferenceOutput
   - Project Dir2
     - ReferenceOutput
   ..

To test the build of the analyzer one would:
   - Copy over a copy of the Repository Directory. (TODO: Prefer to ensure that 
     the build directory does not pollute the repository to min network traffic).
   - Build all projects, until error. Produce logs to report errors.
   - Compare results.  

The files which should be kept around for failure investigations: 
   RepositoryCopy/Project DirI/ScanBuildResults
   RepositoryCopy/Project DirI/run_static_analyzer.log      
   
Assumptions (TODO: shouldn't need to assume these.):   
   The script is being run from the Repository Directory.
   The compiler for scan-build is in the PATH.
   export PATH=/Users/zaks/workspace/c2llvm/build/Release+Asserts/bin:$PATH

For more logging, set the  env variables:
   zaks:TI zaks$ export CCC_ANALYZER_LOG=1
   zaks:TI zaks$ export CCC_ANALYZER_VERBOSE=1
"""
import CmpRuns

import os
import csv
import sys
import glob
import shutil
import time
import plistlib
from subprocess import check_call

# Project map stores info about all the "registered" projects.
ProjectMapFile = "projectMap.csv"

# Names of the project specific scripts.
# The script that needs to be executed before the build can start.
PreprocessScript = "pre_run_static_analyzer.sh"
# This is a file containing commands for scan-build.  
BuildScript = "run_static_analyzer.cmd"

# The log file name.
BuildLogName = "run_static_analyzer.log"
# Summary file - contains the summary of the failures. Ex: This info can be be  
# displayed when buildbot detects a build failure.
NumOfFailuresInSummary = 10
FailuresSummaryFileName = "failures.txt"
# Summary of the result diffs.
DiffsSummaryFileName = "diffs.txt"

# The scan-build result directory.
SBOutputDirName = "ScanBuildResults"
SBOutputDirReferencePrefix = "Ref"

Verbose = 1

def getProjectMapPath():
    ProjectMapPath = os.path.join(os.path.abspath(os.curdir), 
                                  ProjectMapFile)
    if not os.path.exists(ProjectMapPath):
        print "Error: Cannot find the Project Map file " + ProjectMapPath +\
                "\nRunning script for the wrong directory?"
        sys.exit(-1)  
    return ProjectMapPath         

def getProjectDir(ID):
    return os.path.join(os.path.abspath(os.curdir), ID)        

# Run pre-processing script if any.
def runPreProcessingScript(Dir, PBuildLogFile):
    ScriptPath = os.path.join(Dir, PreprocessScript)
    if os.path.exists(ScriptPath):
        try:
            if Verbose == 1:        
                print "  Executing: %s" % (ScriptPath,)
            check_call("chmod +x %s" % ScriptPath, cwd = Dir, 
                                              stderr=PBuildLogFile,
                                              stdout=PBuildLogFile, 
                                              shell=True)    
            check_call(ScriptPath, cwd = Dir, stderr=PBuildLogFile,
                                              stdout=PBuildLogFile, 
                                              shell=True)
        except:
            print "Error: The pre-processing step failed. See ", \
                  PBuildLogFile.name, " for details."
            sys.exit(-1)

# Build the project with scan-build by reading in the commands and 
# prefixing them with the scan-build options.
def runScanBuild(Dir, SBOutputDir, PBuildLogFile):
    BuildScriptPath = os.path.join(Dir, BuildScript)
    if not os.path.exists(BuildScriptPath):
        print "Error: build script is not defined: %s" % BuildScriptPath
        sys.exit(-1)       
    SBOptions = "-plist -o " + SBOutputDir + " "
    SBOptions += "-enable-checker core,deadcode.DeadStores"    
    try:
        SBCommandFile = open(BuildScriptPath, "r")
        SBPrefix = "scan-build " + SBOptions + " "
        for Command in SBCommandFile:
            SBCommand = SBPrefix + Command
            if Verbose == 1:        
                print "  Executing: %s" % (SBCommand,)
            check_call(SBCommand, cwd = Dir, stderr=PBuildLogFile,
                                             stdout=PBuildLogFile, 
                                             shell=True)
    except:
        print "Error: scan-build failed. See ",PBuildLogFile.name,\
              " for details."
        sys.exit(-1)

def buildProject(Dir, SBOutputDir):
    TBegin = time.time() 

    BuildLogPath = os.path.join(Dir, BuildLogName)
    print "Log file: %s" % (BuildLogPath,) 

    # Clean up the log file.
    if (os.path.exists(BuildLogPath)) :
        RmCommand = "rm " + BuildLogPath
        if Verbose == 1:
            print "  Executing: %s." % (RmCommand,)
        check_call(RmCommand, shell=True)
        
    # Open the log file.
    PBuildLogFile = open(BuildLogPath, "wb+")
    try:
        # Clean up scan build results.
        if (os.path.exists(SBOutputDir)) :
            RmCommand = "rm -r " + SBOutputDir
            if Verbose == 1: 
                print "  Executing: %s" % (RmCommand,)
                check_call(RmCommand, stderr=PBuildLogFile, 
                                      stdout=PBuildLogFile, shell=True)
    
        runPreProcessingScript(Dir, PBuildLogFile)
        runScanBuild(Dir, SBOutputDir, PBuildLogFile)        
    finally:
        PBuildLogFile.close()
        
    print "Build complete (time: %.2f). See the log for more details: %s" % \
           ((time.time()-TBegin), BuildLogPath) 
       
# A plist file is created for each call to the analyzer(each source file).
# We are only interested on the once that have bug reports, so delete the rest.        
def CleanUpEmptyPlists(SBOutputDir):
    for F in glob.glob(SBOutputDir + "/*/*.plist"):
        P = os.path.join(SBOutputDir, F)
        
        Data = plistlib.readPlist(P)
        # Delete empty reports.
        if not Data['files']:
            os.remove(P)
            continue

# Given the scan-build output directory, checks if the build failed 
# (by searching for the failures directories). If there are failures, it 
# creates a summary file in the output directory.         
def checkBuild(SBOutputDir):
    # Check if there are failures.
    Failures = glob.glob(SBOutputDir + "/*/failures/*.stderr.txt")
    TotalFailed = len(Failures);
    if TotalFailed == 0:
        CleanUpEmptyPlists(SBOutputDir)
        Plists = glob.glob(SBOutputDir + "/*/*.plist")
        print "Number of bug reports (non empty plist files) produced: %d" %\
           len(Plists)
        return;
    
    # Create summary file to display when the build fails.
    SummaryPath = os.path.join(SBOutputDir, FailuresSummaryFileName);
    if (Verbose > 0):
        print "  Creating the failures summary file %s." % (SummaryPath,)
    
    SummaryLog = open(SummaryPath, "w+")
    try:
        SummaryLog.write("Total of %d failures discovered.\n" % (TotalFailed,))
        if TotalFailed > NumOfFailuresInSummary:
            SummaryLog.write("See the first %d below.\n" 
                                                   % (NumOfFailuresInSummary,))
        # TODO: Add a line "See the results folder for more."
    
        FailuresCopied = NumOfFailuresInSummary
        Idx = 0
        for FailLogPathI in glob.glob(SBOutputDir + "/*/failures/*.stderr.txt"):
            if Idx >= NumOfFailuresInSummary:
                break;
            Idx += 1 
            SummaryLog.write("\n-- Error #%d -----------\n" % (Idx,));
            FailLogI = open(FailLogPathI, "r");
            try: 
                shutil.copyfileobj(FailLogI, SummaryLog);
            finally:
                FailLogI.close()
    finally:
        SummaryLog.close()
    
    print "Error: Scan-build failed. See ", \
          os.path.join(SBOutputDir, FailuresSummaryFileName)
    sys.exit(-1)       

# Auxiliary object to discard stdout.
class Discarder(object):
    def write(self, text):
        pass # do nothing

# Compare the warnings produced by scan-build.
def runCmpResults(Dir):   
    TBegin = time.time() 

    RefDir = os.path.join(Dir, SBOutputDirReferencePrefix + SBOutputDirName)
    NewDir = os.path.join(Dir, SBOutputDirName)
    
    # We have to go one level down the directory tree.
    RefList = glob.glob(RefDir + "/*") 
    NewList = glob.glob(NewDir + "/*")
    if len(RefList) == 0 or len(NewList) == 0:
        return False
    assert(len(RefList) == len(NewList))

    # There might be more then one folder underneath - one per each scan-build 
    # command (Ex: one for configure and one for make).
    if (len(RefList) > 1):
        # Assume that the corresponding folders have the same names.
        RefList.sort()
        NewList.sort()
    
    # Iterate and find the differences.
    HaveDiffs = False
    PairList = zip(RefList, NewList)    
    for P in PairList:    
        RefDir = P[0] 
        NewDir = P[1]
    
        assert(RefDir != NewDir) 
        if Verbose == 1:        
            print "  Comparing Results: %s %s" % (RefDir, NewDir)
    
        DiffsPath = os.path.join(NewDir, DiffsSummaryFileName)
        Opts = CmpRuns.CmpOptions(DiffsPath)
        # Discard everything coming out of stdout (CmpRun produces a lot of them).
        OLD_STDOUT = sys.stdout
        sys.stdout = Discarder()
        # Scan the results, delete empty plist files.
        HaveDiffs = CmpRuns.cmpScanBuildResults(RefDir, NewDir, Opts, False)
        sys.stdout = OLD_STDOUT
        if HaveDiffs:
            print "Warning: difference in diagnostics. See %s" % (DiffsPath,)
            HaveDiffs=True
                    
    print "Diagnostic comparison complete (time: %.2f)." % (time.time()-TBegin) 
    return HaveDiffs

def testProject(ID, IsReferenceBuild, Dir=None):
    TBegin = time.time() 

    if Dir is None :
        Dir = getProjectDir(ID)        
    if Verbose == 1:        
        print "  Build directory: %s." % (Dir,)
    
    # Set the build results directory.
    if IsReferenceBuild == True :
        SBOutputDir = os.path.join(Dir, SBOutputDirReferencePrefix + \
                                        SBOutputDirName)
    else :    
        SBOutputDir = os.path.join(Dir, SBOutputDirName)
    
    buildProject(Dir, SBOutputDir)    

    checkBuild(SBOutputDir)
    
    if IsReferenceBuild == False:
        runCmpResults(Dir)
        
    print "Completed tests for project %s (time: %.2f)." % \
          (ID, (time.time()-TBegin))
    
def testAll(IsReferenceBuild=False):
    PMapFile = open(getProjectMapPath(), "rb")
    try:
        PMapReader = csv.reader(PMapFile)
        for I in PMapReader:
            print " --- Building project %s" % (I[0],)
            testProject(I[0], IsReferenceBuild)            
    finally:
        PMapFile.close()    
            
if __name__ == '__main__':
    testAll()
