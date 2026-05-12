# pocketInformedV-SYNTHES sync

Bidirectional rsync sync between two storage systems for collaborative editing.

- **Local (lab NAS):** `/mnt/katritch_lab2/pocketInformedV-SYNTHES/`
- **Remote (CARC):** `aoxu@discovery.usc.edu:/project2/katritch_223/aoxu/pocketInformedV-SYNTHES/`

Sync is driven **from the local side** (which can SSH out to CARC). CARC runs nothing.

## TL;DR workflow

**You (on lab NAS):**
```
cd /mnt/katritch_lab2/pocketInformedV-SYNTHES
./.sync/sync-both.sh     # pull collaborator's changes
# ...edit, run scripts, generate output...
./.sync/sync-both.sh     # push your changes
```

**Your collaborator (on CARC):** just edits files at `/project2/katritch_223/aoxu/pocketInformedV-SYNTHES/`. They cannot initiate sync themselves (they'd need SSH access back into the lab NAS). They rely on the cron job or on you running `sync-both.sh` manually.

## What's installed

### Scripts (in `.sync/`)
| File | Purpose |
|---|---|
| `sync-both.sh` | Pull then push. The normal entry point. Shared timestamp, `flock` serialization, 100 GB free-space preflight. |
| `sync-pull.sh` | One-way: remote → local (`rsync --update`). |
| `sync-push.sh` | One-way: local → remote (`rsync --update`). |
| `exclude.txt`  | `--exclude-from` list. Skips caches, venvs, `.sync/`, `.claude/`, `.git/`. |
| `README.md`    | This file. |

### Directories (in `.sync/`)
| Dir | Contents |
|---|---|
| `logs/` | Per-run rsync logs: `pull-<ts>.log`, `push-<ts>.log`. Cron also writes `cron.log` here. Auto-pruned after **14 days**. |
| `conflicts/<ts>/from-remote/` | On local: files that were overwritten during pull by a newer remote copy. |
| `conflicts/<ts>/from-local/`  | On remote: files that were overwritten during push by a newer local copy. |
| (pruning) | Empty conflict dirs removed immediately; non-empty ones pruned after **30 days**. |

### Cron (active)
Crontab entry on the local machine:
```
0 */2 * * * /mnt/katritch_lab2/pocketInformedV-SYNTHES/.sync/sync-both.sh >> /mnt/katritch_lab2/pocketInformedV-SYNTHES/.sync/logs/cron.log 2>&1
```
Runs at 00:00, 02:00, 04:00, …, 22:00 local time (12 times/day).

## How bidirectional sync actually works

`rsync` isn't bidirectional on its own. We get the effect by running it **twice per sync cycle** with `--update`:

1. **Pull** (remote → local): files newer on remote are copied down; files newer on local are left alone (`--update`).
2. **Push** (local → remote): files newer on local are copied up; files newer on remote are left alone.

After both steps, each file is at its newest version on both sides. Files that only exist on one side are copied to the other.

### Conflict handling

If the **same file** was edited on both sides between sync runs:
- The newer mtime wins.
- The older version is preserved via `--backup --backup-dir=.sync/conflicts/<ts>/`.
- `<ts>` is shared between the pull and push of one sync run, so both sides' conflict dirs are stamped identically.

Recovering from a conflict:
```
ls .sync/conflicts/                             # list timestamps
ls .sync/conflicts/20260420-173151/from-remote/ # files your local lost to remote
diff path/to/current.txt .sync/conflicts/20260420-173151/from-remote/path/to/current.txt
```

### Deletion behavior

Deletions are **not** propagated (no `--delete`). If you `rm` a file on one side, the next sync restores it from the other side. **To really delete, remove it on both sides.**

If you want deletion propagation later, add `--delete` to the rsync invocations in `sync-pull.sh` / `sync-push.sh` — but understand this turns the remote side into a hostage (anything the collaborator adds between syncs can be wiped out if they're slower than you).

## rsync options used

```
-a                       # archive (perms, times, symlinks, recursive)
-h                       # human-readable
--update                 # skip if destination newer; the core of bidirectional safety
--partial                # keep partial transfers, enable resume
--info=progress2,stats2  # progress + summary on stdout
--exclude-from=exclude.txt
--backup --backup-dir=...
--log-file=...
```

Not used: `--numeric-ids` (local and remote have different UIDs), `--delete` (see above), `--checksum` (slow, mtime comparison is enough with clock-synced USC systems), `-z` (compression not needed on the fast USC network between NAS and CARC).

## Excludes

From `exclude.txt`:
```
.sync/
.claude/
.git/
**/__pycache__/
**/.pytest_cache/
**/.mypy_cache/
**/.ipynb_checkpoints/
**/build/
**/dist/
**/*.egg-info/
**/.venv/
**/venv/
**/wandb/
**/tmp/
**/.DS_Store
```
Edit this file to add more — one glob per line.

## Safety rails

- **Free-space preflight:** aborts if `df` shows less than `MIN_FREE_GB` (default 100) available on the lab NAS. Override per-run: `MIN_FREE_GB=50 ./.sync/sync-both.sh`.
- **Lock file:** `.sync/sync.lock` + `flock` serializes runs. Cron + manual invocation is safe.
- **Pre-flight failure exits** non-zero with a specific code:
  - `2` = not enough disk space
  - `3` = another sync already running
  - `1` = rsync itself failed (check the rsync log in `.sync/logs/`)

## Common commands

| Task | Command |
|---|---|
| Sync now | `./.sync/sync-both.sh` |
| Pull only (don't push your WIP) | `./.sync/sync-pull.sh` |
| Push only (don't pull) | `./.sync/sync-push.sh` |
| Watch the cron log | `tail -f .sync/logs/cron.log` |
| Most recent rsync log | `ls -t .sync/logs/ \| head -5` |
| Change cron schedule | `crontab -e` |
| Disable cron | `crontab -e` and comment the line, OR `crontab -r` to remove all |
| Check if sync is running now | `ls -la .sync/sync.lock` (held by flock if a run is in progress) |
| See conflict history | `find .sync/conflicts -type f` |
| Force-override free-space check | `MIN_FREE_GB=10 ./.sync/sync-both.sh` |

## Changing the cron interval

```
crontab -e
```
Then edit the first two fields (`minute hour`). Examples:

| Every… | Line |
|---|---|
| 15 min | `*/15 * * * * …` |
| 30 min | `*/30 * * * * …` |
| 1 hour | `0 * * * * …` |
| 2 hours (current) | `0 */2 * * * …` |
| 4 hours | `0 */4 * * * …` |
| Business hours only (9am–6pm, hourly) | `0 9-18 * * * …` |

## Troubleshooting

**"PULL FAILED" or "PUSH FAILED" in cron.log**
Check the per-run log: `cat .sync/logs/pull-<latest-ts>.log` (or `push-`). Common causes:
- SSH key unavailable (check `ssh -o BatchMode=yes aoxu@discovery.usc.edu 'hostname'` as your user)
- Remote disk full (check `ssh aoxu@discovery.usc.edu df -h /project2/katritch_223`)
- Network hiccup — rerun; `--partial` resumes in-flight transfers.

**"ABORT: only <N>G free"**
Lab NAS is full. Either free space or override: `MIN_FREE_GB=<lower> ./.sync/sync-both.sh`.

**"ABORT: another sync is already running"**
A previous run is still going. Wait or inspect `ps -ef | grep rsync`.

**Same file keeps flip-flopping between versions**
Clock skew between systems. Check `date` on both; if they differ by >1 min, file a ticket / fix NTP. Workaround: sync more often so whichever you edit most recently wins consistently.

**Files I deleted keep coming back**
Expected — deletions don't propagate. Delete on both sides to really remove.

**Where did my file go?**
Check `.sync/conflicts/<ts>/` — if your local edit lost to a newer remote edit, it's under `from-remote/`.

## Design notes / what wasn't chosen

- **Not unison / syncthing / mutagen** — not installed on CARC, can't install daemons on the lab NAS side either. `rsync` is the only sync tool available on both sides, and it's sufficient for human-paced collaboration.
- **Not git** — the tree is empty now, but likely grows to include large data/model files that don't belong in git. Can be layered on top later for code-only versioning.
- **Not SLURM-driven sync** — the existing `rsync_projects.sbatch` pattern on CARC is designed for bulk one-shot migrations; a periodic bidirectional sync belongs on the side that can SSH out (local machine + cron), not as a SLURM job.
- **Not driven from CARC** — CARC login nodes aren't meant to host cron, and they can't initiate outbound SSH to the lab NAS anyway.
