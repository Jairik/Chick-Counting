#!/bin/bash

# Gets git stats for the repository, including:
# - Number of commits per author
# - Lines added/removed per author
# - Net lines per author
# Note: Specific to this repository, as it includes name mapping
# Usage: ./get-git-stats.sh

# 1. Fetch all branches
git fetch --all

# 2. Produce a log with author markers and numstat
git log main pi \
  --pretty=format:'@@@%aN' \
  --numstat | \
awk '
  # normalize() collapses multiple aliases into a single canonical name
  function normalize(n) {
    if (n == "JJ McCauley" || n == "Jairik" || n == "Jairik McCauley") return "Jairik"
    if (n == "Anye-Nkwenti Forti" || n == "aforti1")      return "Anye-Nkwenti Forti"
    return n
  }

  # CSV header
  BEGIN {
    OFS = ","
    print "Author","Commits","LinesAdded","LinesRemoved","LinesCombined"
  }

  # Lines beginning with @@@<author> mark a new commit
  /^@@@/ {
    raw = substr($0,4)
    author = normalize(raw)
    commits[author]++
    next
  }

  # numstat lines: <added> <removed> <filename>
  /^[0-9]/ {
    added[author]   += $1
    removed[author] += $2
  }

  # At the end, emit CSV rows
  END {
    for (a in commits) {
      la = added[a]   + 0  # ensure zero if undefined
      lr = removed[a] + 0
      lc = la + lr
      print a, commits[a], la, lr, lc
    }
  }
' | sort -t, -k5 -nr

