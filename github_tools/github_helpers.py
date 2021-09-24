"""GitHub GraphQL helpers.

https://developer.github.com/v4/explorer/ is very useful for building queries.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import os
from typing import Dict, Generator, Optional, Tuple

# https://pypi.org/project/gql/
import gql
import gql.transport.requests

_ENV_TOKEN = "GITHUB_ACCESS_TOKEN"

# Query elements for pagination.
PAGINATION = """pageInfo {
  hasNextPage
  endCursor
}
totalCount"""


def add_access_token_arg(
    parser: argparse.ArgumentParser, permissions: str
) -> None:
    """Adds a flag to set the access token."""
    access_token = os.environ.get(_ENV_TOKEN, default=None)
    parser.add_argument(
        "--access-token",
        metavar="ACCESS_TOKEN",
        default=access_token,
        required=not access_token,
        help="The access token for use with GitHub. May also be specified in "
        "the environment as %s. The access token should have permissions: %s"
        % (_ENV_TOKEN, permissions),
    )


class Client(object):
    """A GitHub GraphQL client."""

    def __init__(self, parsed_args: argparse.Namespace):
        """Connects to GitHub."""
        transport = gql.transport.requests.RequestsHTTPTransport(
            url="https://api.github.com/graphql",
            headers={"Authorization": "bearer %s" % parsed_args.access_token},
        )
        self._client = gql.Client(transport=transport)  # type: ignore

    def execute(self, query: str) -> Dict:
        """Runs a query."""
        return self._client.execute(gql.gql(query))  # type: ignore

    def execute_and_paginate(
        self,
        query: str,
        path: Tuple[str, ...],
        first_page: Optional[Dict] = None,
    ) -> Generator[Dict, None, None]:
        """Runs a query with pagination.

        Arguments:
          query: The GraphQL query template, which must have both 'cursor' and
            'pagination' fields to fill in. The cursor should be part of the
            location query (with 'first'), and the pagination should be at the
            same level as nodes.
          path: A list of strings indicating the path to the nodes in the
            result.
          first_page: An optional object for the first page of results, which
            will otherwise automatically be collected. This exists for callers
            to optimize by collecting other data with the first page.
        """
        format = {"cursor": "", "pagination": PAGINATION}
        count = 0
        exp_count = None
        while True:
            if first_page:
                result = first_page
                first_page = None
            else:
                result = self.execute(query % format)
            # Follow the path to the nodes being paginated.
            node_parent = result
            for entry in path:
                node_parent = node_parent[entry]
            # Store the total count of responses.
            if not exp_count:
                exp_count = node_parent["totalCount"]
            # Yield each node individually.
            for node in node_parent["nodes"]:
                yield node
                count += 1
            # Check for pagination, verifying the total count on exit.
            page_info = node_parent["pageInfo"]
            if not page_info["hasNextPage"]:
                assert exp_count == count, "exp %d != actual %d at path %s" % (
                    exp_count,
                    count,
                    path,
                )
                return
            format["cursor"] = ' after: "%s"' % page_info["endCursor"]
