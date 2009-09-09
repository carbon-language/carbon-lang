; RUN: llc < %s -march=c | grep common | grep X

@X = linkonce global i32 5
