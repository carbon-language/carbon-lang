#!/bin/bash -eux

title() {
  (grep "^#" "$1" || echo "${2}") | head -n1 | sed 's/^# //'
}

proposals() {
  cd $(dirname "$0")/../../proposals
  TITLE_PROMPT="        - title: "
  URL_PROMPT="          url: /proposals/"
  for f in $(ls p????.md); do
    NAME=${f%.md}
    TITLE=$(title "${f}" "${NAME}")
    echo -n "${TITLE_PROMPT}${TITLE}^${URL_PROMPT}${NAME}.html^";

    DECISION="${NAME}-decision.md"
    if [ -e "${DECISION}" ]; then
      DECISION_TITLE=$(title "${DECISION}" "${TITLE}: Decision")
      echo -n "${TITLE_PROMPT}${DECISION_TITLE}^${URL_PROMPT}${NAME}-decision.html^";
    fi
  done
}

PROPOSALS="$(proposals)"

cat sidebar.yml | \
  sed "s~^ *- title: AUTO-INSERT PROPOSALS HERE~${PROPOSALS}~" | \
  tr '^' '\n' > \
  .gen-sidebar.yml
