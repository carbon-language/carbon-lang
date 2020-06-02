#!/usr/bin/env python3

"""
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Updates the list of proposals in proposals/README.md.
"""

import os
import re
import sys

if __name__ == '__main__':
  # Read the proposal dir relative to this script.
  proposal_dir = os.path.realpath(os.path.join(os.path.dirname(sys.argv[0]), '../../proposals/'))
  print(proposal_dir)

  # Identify proposal titles in the file list.
  proposals = ['<!-- This list is updated by '
               'src/scripts/pre-commit-proposal-list.py. -->', '']
  error = False
  for file in os.listdir(proposal_dir):
    file_match = re.match(r'^p([0-9]{4})\.md$', file)
    if not file_match:
      continue
    with open(os.path.join(proposal_dir, file)) as f:
      content = f.read()
    title = content.split('\n')[0]
    title_match = re.match(r'^# (.*)$', title)
    if not title_match:
      print('ERROR: %s doesn\'t have a title on the first line.' % file)
      error = True
    proposals.append('- [%s - %s](%s)' % (file_match[1], title_match[1], file))
    decision_file = 'p%s-decision.md' % file_match[1]
    if os.path.exists(os.path.join(proposal_dir, decision_file)):
      proposals.append('  - [Decision](%s)' % decision_file)
  # We print batched errors for usability, but still need to exit with failure.
  if error:
    sys.exit(1)

  # Replace the README content if needed.
  readme_path = os.path.join(proposal_dir, 'README.md')
  with open(readme_path) as f:
    old_content = f.read()
  proposals_re = re.compile(
      r'(.*<!-- proposals -->)(?:.*)(<!-- endproposals -->)',
      re.DOTALL | re.MULTILINE)
  if not proposals_re.match(old_content):
    print('ERROR: proposals/README.md is missing the '
          '<!-- proposals --> ... <!-- endproposals --> marker.')
    sys.exit(1)
  new_content = proposals_re.sub(r'\1\n%s\n\n\2' % '\n'.join(proposals),
                                 old_content)
  if old_content != new_content:
    print('Updating proposals/README.md')
    with open(readme_path, 'w') as f:
      f.write(new_content)
