// RUN: mkdir -p %T/invalid-files
// RUN: clang-apply-replacements %T/invalid-files
//
// Check that the yaml files are *not* deleted after running clang-apply-replacements without remove-change-desc-files.
// RUN: ls %T/Inputs/invalid-files/invalid-files.yaml
