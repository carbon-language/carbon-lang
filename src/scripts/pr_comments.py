# Lint as: python3
"""Figure out comments on a GitHub PR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import sys

# https://pygithub.readthedocs.io/en/latest/introduction.html
from github import Github


def add_to_threads(comment, remap, threads):
    if comment.in_reply_to_id:
        assert comment.in_reply_to_id in remap
        original_id = remap[comment.in_reply_to_id]
        assert comment.id not in remap
        assert original_id in remap
        remap[comment.id] = original_id
        assert original_id in remap
        threads[original_id].append(comment)
    else:
        original_id = comment.id
        assert original_id not in remap
        remap[original_id] = original_id
        assert original_id not in threads
        threads[original_id] = [comment]


def truncate(text, size=25):
    text = re.sub(r"\s+", " ", text)
    if len(text) <= size:
        return text
    else:
        return text[: size - 3] + "..."


def print_comment(comment):
    print(comment.html_url)
    print(
        " - %s by %s: %s"
        % (comment.created_at, comment.user.login, truncate(comment.body))
    )


def main(argv):
    if len(argv) != 2:
        print("Expected: %s <PR#>" % (argv[0],))
        sys.exit(1)
    pr_num = int(argv[1])
    g = Github(os.environ["GITHUB_ACCESS_TOKEN"])
    repo = g.get_repo("carbon-language/carbon-lang")
    pr = repo.get_pull(pr_num)
    print("PR# %s: %s" % (pr.number, pr.title))
    print()
    print("Reviews")
    for review in pr.get_reviews():
        if review.body:
            suffix = ": %s" % (truncate(review.body),)
        else:
            suffix = ""
        print("- %s by %s%s" % (review.submitted_at, review.user.login, suffix))
    print()
    print("comments: %s" % (pr.comments))
    for comment in pr.get_issue_comments():
        print_comment(comment)
    print()
    print("review_comments: %s" % (pr.review_comments))
    remap = {}
    threads = collections.OrderedDict()
    for comment in pr.get_review_comments():
        add_to_threads(comment, remap, threads)
    for t in threads.values():
        print(t[0].html_url)
        for c in t:
            print(
                " - %s by %s: %s"
                % (c.created_at, c.user.login, truncate(c.body))
            )
        print()


if __name__ == "__main__":
    main(sys.argv)
