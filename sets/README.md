# Eval sets

One folder per set (from 0.1.1). Media is downloaded or copied in locally and never
committed; what is committed is the list of cases, the labels, and where every file came from.

```
sets/<set-name>/
  SOURCES.md     one line per source file: id, origin, licence, date fetched
  cases.jsonl    one event per line: clip id, time, label, box, ground truth per check
  media/         the clips (git-ignored)
```

A file without a `SOURCES.md` entry is not used. File names are neutral ids (`clip-001.mp4`),
never a site, client or employer name.
