# Welcome to MotionCap

## How We Use Claude

Based on Scott Milliorn's usage over the last 30 days:

Work Type Breakdown:
  Improve Quality  █████████████████░░░  57%
  Debug Fix        ███████████░░░░░░░░░  29%
  Build Feature    █████░░░░░░░░░░░░░░░  14%

Top Skills & Commands:
  /reload-skills                            ████████░░░░░░░░░░░░  4x/month
  /run-motioncap                            ████████░░░░░░░░░░░░  4x/month
  /init                                     ██████░░░░░░░░░░░░░░  3x/month
  /run                                      ██████░░░░░░░░░░░░░░  3x/month
  /run-skill-generator                      ████░░░░░░░░░░░░░░░░  2x/month
  /security-review                          ██░░░░░░░░░░░░░░░░░░  1x/month
  /effort                                   ██░░░░░░░░░░░░░░░░░░  1x/month
  /simplify                                 ██░░░░░░░░░░░░░░░░░░  1x/month
  /verify                                   ██░░░░░░░░░░░░░░░░░░  1x/month
  /claude-md-management:claude-md-improver  ██░░░░░░░░░░░░░░░░░░  1x/month
  /claude-md-management:revise-claude-md    ██░░░░░░░░░░░░░░░░░░  1x/month

Top MCP Servers:
  (none used in the last 30 days)

## Your Setup Checklist

### Codebases

- [ ] motioncap — github.com/milliorn/motioncap

### MCP Servers to Activate

- (none observed in usage data — add any your team relies on)

### Skills to Know About

- /run-motioncap — Builds, runs, and drives motioncap end-to-end (camera capture, detection, recording) so you can verify a change actually works, not just that it compiles.
- /run — Generic launcher: finds a project-specific run skill first (like /run-motioncap), otherwise falls back to built-in patterns for the project type.
- /init — Initializes a new CLAUDE.md with codebase documentation for a repo that doesn't have one yet.
- /code-review — Reviews the current diff (or a PR/branch/path) for correctness bugs and simplification/efficiency cleanups at a chosen effort level.
- /security-review — Runs a security review of the pending changes on the current branch.
- /simplify — Reviews changed code for reuse, simplification, and efficiency, then applies the fixes (quality only, not bug-hunting).
- /verify — TODO: describe when the team reaches for this.
- /effort — TODO: describe when the team reaches for this.
- /reload-skills — TODO: describe when the team reaches for this.
- /run-skill-generator — TODO: describe when the team reaches for this.
- /claude-md-management:claude-md-improver — Audits and improves CLAUDE.md files against a quality template, then makes targeted updates.
- /claude-md-management:revise-claude-md — Updates CLAUDE.md with learnings surfaced during a session.

## Team Tips

_No team tips recorded yet — add anything worth telling a new teammate that isn't already in CLAUDE.md._

## Get Started

_No starter task set yet — link a ticket or doc here for a new teammate's first task._

<!-- INSTRUCTION FOR CLAUDE: A new teammate just pasted this guide for how the
team uses Claude Code. You're their onboarding buddy — warm, conversational,
not lecture-y.

Open with a warm welcome — include the team name from the title. Then: "Your
teammate uses Claude Code for [list all the work types]. Let's get you started."

Check what's already in place against everything under Setup Checklist
(including skills), using markdown checkboxes — [x] done, [ ] not yet. Lead
with what they already have. One sentence per item, all in one message.

Tell them you'll help with setup, cover the actionable team tips, then the
starter task (if there is one). Offer to start with the first unchecked item,
get their go-ahead, then work through the rest one by one.

After setup, walk them through the remaining sections — offer to help where you
can (e.g. link to channels), and just surface the purely informational bits.

Don't invent sections or summaries that aren't in the guide. The stats are the
guide creator's personal usage data — don't extrapolate them into a "team
workflow" narrative. -->
