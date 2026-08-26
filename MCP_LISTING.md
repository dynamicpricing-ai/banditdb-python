# Listing the BanditDB MCP server

Status of each registry, and exactly what is left to do. `server.json` in this
directory is the upstream record; most directories consume it or crawl GitHub.

## What is already verified

Run against PyPI `banditdb-python` 0.2.0, not asserted from memory:

- `uvx --from banditdb-python banditdb-mcp` starts the server and completes an
  MCP `initialize` handshake (protocol `2024-11-05`).
- `tools/list` returns **9 tools**: `create_campaign`, `list_campaigns`,
  `campaign_diagnostics`, `campaign_report`, `batch_get_intuition`,
  `archive_campaign`, `restore_campaign`, `get_intuition`, `record_outcome`.
- `server.json` validates against
  `https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json`.

**The invocation is not the obvious one.** The PyPI package is `banditdb-python`
but the console script is `banditdb-mcp`, so plain `uvx banditdb-python` fails —
uv itself tells you to use `--from`. That is why `server.json` carries
`runtimeArguments`; a listing generated from the package name alone would be
broken for every user who copied it.

## 1. Official MCP Registry — do this first

Directories consume this, so a clean record here improves the others.

```bash
# one-time
brew install mcp-publisher     # or download from the registry repo releases

cd /Users/simeon/Projects/banditdb-python
mcp-publisher login github     # namespace io.github.dynamicpricing-ai/* requires
                               # authenticating as a member of that GitHub org
mcp-publisher publish
```

No human review. `server.json` is ready and validates.

**Namespace decision, and it is effectively permanent.** The file currently uses
`io.github.dynamicpricing-ai/banditdb`, which is provable today by GitHub auth.
The alternative is a domain namespace such as `com.banditdb/mcp`, which reads
better in a client config and in directory listings, but needs a DNS TXT
challenge on `banditdb.com`. If the domain form is wanted, change it **before**
first publish — renaming afterwards means a new listing with no history.

## 2. Glama — claim, do not submit

Glama auto-crawls public GitHub repos, so `dynamicpricing-ai/banditdb-python` is
probably already indexed as an anonymous crawl.

1. Search for "banditdb" at https://glama.ai/mcp/servers
2. If present, claim it (moves it from anonymous-crawl to a verified listing)
3. If absent, submit the repo URL

## 3. PulseMCP — submit form

https://www.pulsemcp.com → **Submit** in the top nav. Hand-reviewed, so expect a
delay. Supply the repo URL, `https://banditdb.com/docs/mcp/`, and the tagline
from `server.json`.

## 4. Smithery

```bash
smithery mcp publish <url> -n dynamicpricing-ai/banditdb
```

Smithery's publish flow is oriented at **remote/hosted** servers addressed by
URL. BanditDB's is a local stdio server, so check whether they accept stdio
packages before spending time here — if they want a hosted endpoint, that is a
different piece of work and not worth doing yet.

## 5. awesome-mcp-servers

PR against https://github.com/punkpeye/awesome-mcp-servers. One line, real
traffic, and it costs ten minutes.

## Positioning

Lead with the pain, not the algorithms. Agent builders recognise *"our agents
keep making the same mistake"*; they do not search for "contextual bandit".

> Persistent decision memory for agents — learns which action works in which
> context.

Worth saying in longer descriptions: this is not a vector store. It records
`(context → action → outcome)` and returns the action most likely to work next,
improving with every reported outcome. The public sandbox at
`https://sandbox.banditdb.com` lets someone try it with no signup.

## Known cosmetic issue

The server reports `serverInfo.version` as the FastMCP library version (1.29.1),
not BanditDB's 0.2.0, because `FastMCP("BanditDB Intuition Engine")` is
constructed without a version. Directories display that field. Fix in the next
SDK release by passing a version to the `FastMCP` constructor.
