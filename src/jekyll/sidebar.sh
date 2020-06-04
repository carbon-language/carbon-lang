#!/bin/bash -eux

title() {
  (grep -m1 "^# " "$1" || echo "${2}") | cut -c3-
}

proposals() {
  cd ../../proposals
  TITLE_PROMPT="        - title: "
  URL_PROMPT="          url: /proposals/"
  for f in $(ls p????.md); do
    NAME=${f%.md}
    TITLE=$(title "${f}" "${NAME}")
    echo -n "${TITLE_PROMPT}${TITLE}^${URL_PROMPT}${NAME}.html^";

    DECISION="${NAME}-decision.md"
    if [ -e "../../proposals/${DECISION}" ]; then
      DECISION_TITLE=$(title "${DECISION}" "${TITLE}: Decision")
      echo -n "${TITLE_PROMPT}${DECISION_TITLE}^${URL_PROMPT}${NAME}-decision.html^";
    fi
  done
}

PROPOSALS="$(proposals)"

INSERT_PATTERN="^ *- title: AUTO-INSERT PROPOSALS HERE"
# Print the lines of the input file until the insertion marker (but not including it).
sed -n "/$INSERT_PATTERN/!p;//q" sidebar.yml > .gen-sidebar.yml
# Print the proposals by scanning the files.
proposals >> .gen-sidebar.yml
# Print the lines of the input file after the insertion marker.
sed -n "1,/$INSERT_PATTERN/!p" sidebar.yml >> .gen-sidebar.yml
