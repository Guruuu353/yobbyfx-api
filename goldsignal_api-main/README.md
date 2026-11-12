# goldsignal_api

## v4 changes

- **Scalp mode** follows the 1H (falls back to the 4H) instead of only firing when 1H and 4H disagree. Against the 4H it needs one extra confirmation. It may also use the entry timeframe's own order blocks / gaps.
- **Entry sensitivity** (`sens`: strict / normal / high) in Profile, live scanner, replay and backtest. `strict` = old v3 rules. Switches live in `strategy.engine_flags()`.
- **Catch-up + background scanner**: every closed candle since the last scan is evaluated (`run_scan`), signals that had already resolved are stored silently, and `bg_scanner.py` keeps scanning for 3 days after the last page visit (`GOLDSIGNAL_BG_SCAN=0` disables). A `setup` alert fires when a zone is tapped and a trigger is pending.
- **Backtest = live engine**: same flags, daily/weekly levels, `trades.replay` rules (stop wins a both-hit candle), entry on the next candle's open, spread charged in R.
- **Candle store** (`cache/candles.db`, `candle_store.py`): every closed candle is kept, history grows beyond Yahoo's window. `clean_candles()` aligns, fills holes, fixes OHLC and snaps tiny open gaps on <=30m. `merge_live` no longer shrinks candles.
- **Chart**: scroll back loads older pages (`/market/candles?before=`), `/signals/replay` draws past signals, outcomes and near-misses.

## v5 notes - fewer losses, news protection

**Losses / small wins.** The old engine took any setup with a wall 1.0R away (target ~1.1R), scalps aimed at 1.5R, and after a
stop-out the same idea re-entered a few candles later. Now:
* **Room gate** - the next real wall (1H/4H zone, liquidity pool, daily/weekly level) must be >= 1.5R away (1.2R for short R:R); otherwise the status is `no_room` instead of a tiny win.
* **Scalp target 2.0R** (was 1.5R). **Chop gate** (efficiency ratio over 40 bars < 0.07 = no entries). **Cost gate** (spread must be <= 0.15R of the stop).
* **Cool-down** - after a stop-out no new entry in the same direction for ~6 bars (live tracker and backtest).
* Backtest page has **"Why trades lose"**: win rate needed to break even, losers that were once +0.5R, weak buckets (agreement / pattern / zone / session) and hints.
* Numbers on the bundled demo data are synthetic (random walk): they check the mechanics, not profitability. Judge it with a backtest on real Yahoo data.

**News.** `app/news.py` + `/news/*`: ForexFactory weekly calendar (this + next week, refreshed every 30 min, stored in `news_events`),
a built-in schedule for NFP / FOMC / weekly claims (so backtests over past weeks still see them), manual events, RSS headlines.
`GOLDSIGNAL_NEWS=0` disables the refresher. Hard **news** gate: no new signals from 30 min before to 20 min after a high-impact release for the
pair's currencies (CPI / NFP / FOMC / rate decisions x1.5). Settings: News page -> News protection. The same windows feed the live scan,
the chart replay and the backtest. Alert "News: ..." 15 min before a release. Charts: the **News** layer (line, blackout band, price reaction).

## v6 - installable PWA + background push

The frontend is a full PWA: `manifest.webmanifest`, icons (incl. maskable + Apple touch), `sw.js` (offline app shell, "new version ready" banner;
market data is never cached), install button (top bar + Profile -> "Install as an app", iPhone steps included).

**Install needs https** (or `localhost` on the same computer):
* Deploy: frontend on Vercel (`frontend/`, `npm run build`, env `VITE_API_URL=https://<your-backend>`), backend on Railway / a VPS
  (`uvicorn app.main:app --host 0.0.0.0 --port $PORT`). Mount a volume for `goldsignal.db` and `cache/` (SQLite is wiped on redeploy otherwise).
* Try it on a phone from your PC: `cloudflared tunnel --url http://localhost:5173` (or ngrok) gives an https URL.

**Background alerts:** Profile -> Install as an app -> "Background alerts on this device". The backend encrypts and sends Web Push itself
(`app/push.py`, no extra package) for every alert: signals, TP/SL, setups, news. The VAPID key is created on first start in `backend/vapid_private.pem`;
on hosts with a wiped disk set `GOLDSIGNAL_VAPID_PEM` (PEM text, newlines as `\n`) so subscriptions survive redeploys.
`GOLDSIGNAL_PUSH_SUBJECT=mailto:you@yourdomain.com` sets the contact; `GOLDSIGNAL_PUSH=0` turns sending off. iPhone: only after "Add to Home Screen" (iOS 16.4+).
The server keeps scanning with the browser closed (`bg_scanner.py`), so alerts keep coming while the backend runs.

## v7 - AMD (accumulation -> manipulation -> distribution) + loss minimisation

* `app/amd.py` - works on ANY timeframe, point-in-time (no look-ahead): a tight range (<= 3.2 ATR, low efficiency, both edges respected) ->
  a raid of one edge that closes back inside within 3 bars (the trap) -> displacement through the range midpoint (distribution).
  Chart layer **AMD** draws the range, the raid (M) and the displacement (D) for the timeframe you are viewing; the Log panel shows the current phase.
* Engine switch **AMD**: off / confirm (an aligned sequence is an extra confirmation, a counter-sequence needs one more) / required (only aligned setups).
* Engine switch **Quality**: standard / selective (1H discount-premium location and momentum become hard gates, agreement +5).
* **Daily stop**: after N stop-outs on a pair the engine pauses it for the rest of the UTC day (live and backtest).
* Backtest -> **Compare settings**: six engine settings on the same candles, costs and news windows, plus the last 30% of the period judged separately
  (out-of-sample). The best one is the highest net R per drawdown among settings that also made money in the last 30%; "Use live" applies it.
  If nothing profits it says so - that is information about the pair / timeframe / style, not a bug.
* Settings are saved server-side (`settings.json`): `GET/PUT /engine/settings`.

## v8 - Alerts board, previews, invites, manual

* **Alerts page + bell**: `GET /signals/board` (what every pair is doing, from the scanner's latest evaluation), `POST /signals/board/scan`
  (evaluate pairs now, 3 at a time from the page), `GET /signals/activity` (state changes: watch -> setup -> signal), `GET /signals/peek`
  (read-only analysis of any pair for the pop-up chart). The scan log now stores a small `snapshot_json` (zone, plan, stage) per row.
* **Previews**: price line with entry / stop / target, the fired candle and the outcome on signal cards, open trades and the Signals page.
* **Full-chart pop-up** from any board row, alert, log line, signal or trade.
* **Invites**: `users.invited_by`, `GET /profile/invite`, `?ref=<username>` on `/register`.
* **Manual**: `/guide` (signed in) and `/manual` (public). Chart gained premium/discount + EQ, strong/weak highs & lows, FVG / sweeps / trend lines on 4H,
  Asia / London range boxes, entry sweep, volume spike, retest entry and breakeven lines.
* New columns are added automatically to an existing `goldsignal.db` on start.


## v3: Expert / MetaTrader 5 bridge, always-on scanning, bigger brain

**Profile -> Expert** (in the app) controls everything:
* **Off / Manual / Auto.** Manual: every signal for an allowed pair waits for your "Send to MT5" tap (Profile or the signal card). Auto: it is queued the moment the engine fires.
* **Allowed pairs**: only ticked pairs are scanned 24/5 on the server (even with the browser closed, aligned to each candle close) and traded.
* Filters (grade, swing/scalp/counter-trend, min brain score, max signal age), lots by risk % or fixed, max open, daily loss stop in R, break-even, spread/slippage, broker symbol suffix/overrides, demo-only lock.
* **Linking MT5**: press *Create token*, *Download EA* (token + server address already inside), put `GoldSignalBridge.mq5` in `MQL5/Experts`, compile, allow WebRequest for your server address, attach to any chart. Source: `goldsignal_api-main/ea/GoldSignalBridge.mq5`. Your broker password never leaves your PC. The EA also uploads your terminal's candle history so the brain learns from years instead of Yahoo's 60 days.
* The EA was NOT compiled or run against a real MT5 terminal when written. Compile it, fix any message MetaEditor shows, and test on a DEMO account first.

Server settings: `GOLDSIGNAL_JWT_SECRET` (set it on your host; otherwise a random secret is generated into `.jwt_secret` - everyone logs in once again after this upgrade), `GOLDSIGNAL_EXPERT_SECONDS` (30), `GOLDSIGNAL_BRAIN_BARS` (60000).


## v12 - live MT5 feed, early entries from the brain, live trade management (EA 2.0)

**Problem this fixes.** Yahoo's tape runs minutes behind, so every candle-close signal arrived late: orders aged out (`max_age_bars`, `expire_min`)
or were placed after price had gone, and "sent" orders that the EA could not place just sat there until they expired.

* **Live feed (`app/live_feed.py`).** The EA pushes the broker's live quote + last 1m candles (forming bar included) of every allowed pair on each
  poll (2 s). `data_fetcher` lays them over Yahoo's 1m tape, so every timeframe, the brain and the signal engine run on broker data. Candles are
  treated as closed 3 s after the bar ends (15 s on Yahoo). If the EA goes quiet, everything falls back to Yahoo by itself.
* **Early entries (`app/autopilot.py`, Profile -> Expert -> Early entries).** When the brain reads a side as *armed / ready* (price in the 1H zone,
  trigger candle forming or closed, trend done, no gate closed, readiness >= 80 %, score >= 55) the bridge places a reduced-size trade (default
  0.5x risk) on the live MT5 price, stop/target from the brain's plan. When the engine's real signal fires, the trade is *confirmed*: stop aligned to the
  signal (never widened by more than 25 %) and the rest of the risk is added as a top-up order. If the signal never comes the manager closes it
  (brain readiness fell below 30 %, or N candles passed) - in profit it just goes to break-even.
* **Trade management on every poll.** Trailing stop (from 1.5R, 1R behind the peak), close when the opposite side turns ready (>= 85 %) while the
  trade is under +1R, news guard (break-even or close N minutes before a high-impact release), daily loss stop in % of balance. Commands travel to
  the EA as `MODIFY|id|ticket|sl|tp` and `CLOSE|id|ticket`.
* **Orders that "do not reflect in MT5".** The EA now keeps a local retry queue (every second until placed or expired) and tells the app *why* it
  is waiting (spread above limit, Algo Trading off, requote ...); the server re-offers an unconfirmed order after 10 s instead of 90 s and no longer
  gives up after 5 deliveries. A late signal is skipped on the live price when it has already moved more than `max_drift_r` (default 0.5R) either way.
* **Account / positions.** The EA reports balance, equity, floating P/L, margin and every open position; they show under *Live from MT5*.
* **Clock.** The EA reports the terminal's last-tick time; the server compares it with its own UTC clock and derives the broker's real UTC offset
  (independent of the VPS time zone). The chart's "MT5 server" clock uses it. In `Profile` there is also a fixed "Nairobi (GMT+3)" clock.
  A new candle anchor `fixed` (constant broker offset, no DST) exists for brokers that are not on New York + 7 h; *Match my MT5 clock* applies it.

Environment: `GOLDSIGNAL_EARLY_SECONDS` (10) how often an early-entry pair's brain is re-read.
The EA is not compiled here - recompile `GoldSignalBridge.mq5` in MetaEditor and test on a demo account.
