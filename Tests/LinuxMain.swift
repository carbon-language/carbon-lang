import XCTest

import barconTests

var tests = [XCTestCaseEntry]()
tests += barconTests.allTests()
XCTMain(tests)
