#!/usr/bin/env python

"""
Static Analyzer qualification infrastructure: adding a new project to 
the Repository Directory.

 Add a new project for testing: build it and add to the Project Map file.
   Assumes it's being run from the Repository Directory.
   The project directory should be added inside the Repository Directory and 
   have the same name as the project ID
   
 The project should use the following files for set up:
      - pre_run_static_analyzer.sh - prepare the build environment.
                                     Ex: make clean can be a part of it.
      - run_static_analyzer.cmd - a list of commands to run through scan-build.
                                     Each command should be on a separate line.
                                     Choose from: configure, make, xcodebuild 
"""
import SATestBuild

import os
import csv
import sys

# Add a new project for testing: build it and add to the Project Map file.
# Params:
#   Dir is the directory where the sources are.
#   ID is a short string used to identify a project.
def addNewProject(ID) :
    CurDir = os.path.abspath(os.curdir)
    Dir = SATestBuild.getProjectDir(ID)
    if not os.path.exists(Dir):
        print "Error: Project directory is missing: %s" % Dir
        sys.exit(-1)
        
    # Build the project.
    SATestBuild.testProject(ID, True, Dir)

    # Add the project ID to the project map.
    ProjectMapPath = os.path.join(CurDir, SATestBuild.ProjectMapFile)
    if os.path.exists(ProjectMapPath):
        PMapFile = open(ProjectMapPath, "r+b")
    else:
        print "Warning: Creating the Project Map file!!"
        PMapFile = open(ProjectMapPath, "w+b")
    try:
        PMapReader = csv.reader(PMapFile)
        for I in PMapReader:
            IID = I[0]
            if ID == IID:
                print >> sys.stderr, 'Warning: Project with ID \'', ID, \
                        '\' already exists.'
                sys.exit(-1)

        PMapWriter = csv.writer(PMapFile)
        PMapWriter.writerow( (ID, Dir) );
    finally:
        PMapFile.close()
        
    print "The project map is updated: ", ProjectMapPath
            

# TODO: Add an option not to build. 
# TODO: Set the path to the Repository directory.
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print >> sys.stderr, 'Usage: ', sys.argv[0],\
                             '[project ID]'
        sys.exit(-1)
        
    addNewProject(sys.argv[1])
