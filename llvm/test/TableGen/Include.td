// RUN: tblgen -I `dirname %s` %s
def BeforeInclude;

include "Include.inc"

def AfterInclude;

