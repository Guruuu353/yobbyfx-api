//+------------------------------------------------------------------+
//|  GoldSignalBridge.mq5  -  version 1.0                             |
//|  Places the signals from your GoldSignal app in this MT5 terminal |
//|                                                                    |
//|  HOW IT WORKS                                                      |
//|    every few seconds the EA asks your GoldSignal server "is there  |
//|    an order for me?".  If you chose Auto (or pressed Send to MT5)  |
//|    it places the trade at the live broker price, then reports the  |
//|    fill / rejection / close back to the app.                       |
//|    Your broker password never leaves this PC.                      |
//|                                                                    |
//|  SET-UP (once)                                                     |
//|    1. File > Open Data Folder > MQL5 > Experts : put this file     |
//|       there, open it in MetaEditor and press Compile (F7).         |
//|    2. In MT5: Tools > Options > Expert Advisors:                   |
//|         [x] Allow algorithmic trading                              |
//|         [x] Allow WebRequest for listed URL  -> add the address    |
//|             shown in InpApiUrl below (only the https://host part). |
//|    3. Drag "GoldSignalBridge" onto ANY chart, press the            |
//|       "Algo Trading" button so it is green.  Done: the app's       |
//|       Expert card turns to "Connected".                            |
//|    The EA refuses REAL accounts unless you also tick               |
//|    InpAllowLiveAccount below AND untick "Demo only" in the app.    |
//+------------------------------------------------------------------+
#property copyright "GoldSignal"
#property version   "1.00"
#property description "GoldSignal MT5 bridge - executes the signals approved in the app"

#include <Trade\Trade.mqh>

#define GS_VERSION "1.0"

input string InpToken            = "__TOKEN__";     // Bridge token (already filled in by the app)
input string InpApiUrl           = "__API_URL__";   // Your GoldSignal backend address
input bool   InpAllowLiveAccount = false;           // Allow trading on a REAL account (default: demo only)
input bool   InpUploadHistory    = true;            // Send this terminal's candle history to the app (brain learns from it)
input int    InpHistoryBars      = 30000;           // Candles per pair and timeframe to send the first time

CTrade   g_trade;
string   g_cfgKeys[];
string   g_cfgVals[];
string   g_pairs[];
ulong    g_lastPollMs   = 0;
string   g_status       = "starting";
bool     g_historyBuilt = false;

// history upload state machine (one chunk per timer tick, so polling is never blocked for long)
string          g_hSym[];
ENUM_TIMEFRAMES g_hTf[];
string          g_hTfName[];
int             g_hIdx   = 0;
int             g_hOff   = 0;
int             g_hWant  = 0;
int             g_hFails = 0;

// positions opened by this EA that we still have to report as closed
long     g_trkId[];
ulong    g_trkTicket[];

//+------------------------------------------------------------------+
//| helpers                                                           |
//+------------------------------------------------------------------+
string JsonEsc(string s)
{
   StringReplace(s, "\\", "\\\\");
   StringReplace(s, "\"", "\\\"");
   StringReplace(s, "\r", " ");
   StringReplace(s, "\n", " ");
   StringReplace(s, "\t", " ");
   return s;
}

string CfgGet(string key, string def)
{
   int n = ArraySize(g_cfgKeys);
   for(int i = 0; i < n; i++)
      if(g_cfgKeys[i] == key)
         return g_cfgVals[i];
   return def;
}

double CfgNum(string key, double def)
{
   string v = CfgGet(key, "");
   if(v == "")
      return def;
   return StringToDouble(v);
}

string ServerBase()
{
   string u = InpApiUrl;
   while(StringLen(u) > 0 && StringSubstr(u, StringLen(u) - 1, 1) == "/")
      u = StringSubstr(u, 0, StringLen(u) - 1);
   return u;
}

// POST json to the app. Returns the HTTP status (or -1 when the request could not be sent)
int Post(string path, string body, string &response)
{
   char   data[];
   char   result[];
   string resHeaders;
   string headers = "Content-Type: application/json\r\nX-Bridge-Token: " + InpToken + "\r\n";
   int len = StringToCharArray(body, data, 0, WHOLE_ARRAY, CP_UTF8);
   if(len > 0)
      ArrayResize(data, len - 1);        // drop the terminating zero
   ResetLastError();
   int code = WebRequest("POST", ServerBase() + path, headers, 8000, data, result, resHeaders);
   if(code == -1)
   {
      int err = GetLastError();
      if(err == 4014)
         g_status = "WebRequest blocked: add " + ServerBase() + " in Tools > Options > Expert Advisors";
      else
         g_status = "cannot reach the server (error " + IntegerToString(err) + ")";
      response = "";
      return -1;
   }
   response = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
   return code;
}

bool IsRealAccount()
{
   return (ENUM_ACCOUNT_TRADE_MODE)AccountInfoInteger(ACCOUNT_TRADE_MODE) == ACCOUNT_TRADE_MODE_REAL;
}

string AccountJson()
{
   bool real = IsRealAccount();
   string s = "{";
   s += "\"login\":" + IntegerToString((long)AccountInfoInteger(ACCOUNT_LOGIN)) + ",";
   s += "\"server\":\"" + JsonEsc(AccountInfoString(ACCOUNT_SERVER)) + "\",";
   s += "\"company\":\"" + JsonEsc(AccountInfoString(ACCOUNT_COMPANY)) + "\",";
   s += "\"name\":\"" + JsonEsc(AccountInfoString(ACCOUNT_NAME)) + "\",";
   s += "\"currency\":\"" + JsonEsc(AccountInfoString(ACCOUNT_CURRENCY)) + "\",";
   s += "\"balance\":" + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + ",";
   s += "\"equity\":" + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + ",";
   s += "\"margin_free\":" + DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN_FREE), 2) + ",";
   s += "\"leverage\":" + IntegerToString((long)AccountInfoInteger(ACCOUNT_LEVERAGE)) + ",";
   s += "\"trade_mode\":" + IntegerToString((long)AccountInfoInteger(ACCOUNT_TRADE_MODE)) + ",";
   s += "\"demo\":" + (real ? "false" : "true") + ",";
   s += "\"build\":" + IntegerToString((long)TerminalInfoInteger(TERMINAL_BUILD)) + ",";
   s += "\"terminal_trade_allowed\":" + (TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) ? "true" : "false");
   s += "}";
   return s;
}

void Report(long id, string ev, ulong ticket, double price, double lots, double profit, string err)
{
   string b = "{\"id\":" + IntegerToString(id) + ",\"event\":\"" + ev + "\"";
   if(ticket > 0)
      b += ",\"ticket\":\"" + IntegerToString((long)ticket) + "\"";
   if(price > 0)
      b += ",\"price\":" + DoubleToString(price, 8);
   if(lots > 0)
      b += ",\"lots\":" + DoubleToString(lots, 4);
   if(ev == "closed")
      b += ",\"profit\":" + DoubleToString(profit, 2);
   if(err != "")
      b += ",\"error\":\"" + JsonEsc(err) + "\"";
   b += "}";
   string resp;
   int code = -1;
   for(int k = 0; k < 3 && code != 200; k++)       // a lost report would leave the app guessing: try a few times
      code = Post("/bridge/ea/report", b, resp);
}

string Tag(long id)
{
   return "GS#" + IntegerToString(id);
}

//+------------------------------------------------------------------+
//| tracking of positions we opened                                   |
//+------------------------------------------------------------------+
void TrackAdd(long id, ulong ticket)
{
   int n = ArraySize(g_trkId);
   for(int i = 0; i < n; i++)
      if(g_trkId[i] == id)
         return;
   ArrayResize(g_trkId, n + 1);
   ArrayResize(g_trkTicket, n + 1);
   g_trkId[n] = id;
   g_trkTicket[n] = ticket;
   GlobalVariableSet("GSB_OPEN_" + IntegerToString(id), (double)ticket);
}

void TrackRemoveAt(int idx)
{
   int n = ArraySize(g_trkId);
   if(idx < 0 || idx >= n)
      return;
   GlobalVariableDel("GSB_OPEN_" + IntegerToString(g_trkId[idx]));
   for(int i = idx; i < n - 1; i++)
   {
      g_trkId[i] = g_trkId[i + 1];
      g_trkTicket[i] = g_trkTicket[i + 1];
   }
   ArrayResize(g_trkId, n - 1);
   ArrayResize(g_trkTicket, n - 1);
}

void TrackReload()
{
   ArrayResize(g_trkId, 0);
   ArrayResize(g_trkTicket, 0);
   int total = GlobalVariablesTotal();
   for(int i = 0; i < total; i++)
   {
      string nm = GlobalVariableName(i);
      if(StringFind(nm, "GSB_OPEN_") == 0)
      {
         long id = StringToInteger(StringSubstr(nm, 9));
         ulong tk = (ulong)GlobalVariableGet(nm);
         int n = ArraySize(g_trkId);
         ArrayResize(g_trkId, n + 1);
         ArrayResize(g_trkTicket, n + 1);
         g_trkId[n] = id;
         g_trkTicket[n] = tk;
      }
   }
}

// find an open position that carries this order's tag (guards against placing the same order twice)
ulong FindPositionByTag(long id, long magic)
{
   string tag = Tag(id);
   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != magic)
         continue;
      string c = PositionGetString(POSITION_COMMENT);
      if(StringFind(c, tag) == 0 && (StringLen(c) == StringLen(tag) || StringSubstr(c, StringLen(tag), 1) == " "))
         return tk;
   }
   return 0;
}

//+------------------------------------------------------------------+
//| sizing                                                            |
//+------------------------------------------------------------------+
double NormalizeLots(string sym, double lots)
{
   double step = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
   if(step <= 0)
      step = 0.01;
   lots = MathFloor(lots / step + 1e-9) * step;
   if(lots < vmin)
      lots = vmin;
   if(lots > vmax)
      lots = vmax;
   return NormalizeDouble(lots, 2);
}

// money lost per 1.0 lot if the stop is hit
double LossPerLot(string sym, bool isBuy, double price, double sl)
{
   double profit = 0.0;
   ENUM_ORDER_TYPE ot = isBuy ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   if(OrderCalcProfit(ot, sym, 1.0, price, sl, profit) && profit < 0.0)
      return -profit;
   double tickSize  = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   if(tickSize > 0 && tickValue > 0)
      return MathAbs(price - sl) / tickSize * tickValue;
   return 0.0;
}

//+------------------------------------------------------------------+
//| open an order from the app.  Returns true = handled (do not       |
//| fetch again), false = try again on a later poll                   |
//+------------------------------------------------------------------+
bool ExecuteOpen(string &p[])
{
   // OPEN|id|SYMBOL|BUY/SELL|entry|sl|tp|sl_dist|tp_dist|lots|risk_pct|expires_ts
   if(ArraySize(p) < 12)
      return true;
   long   id        = StringToInteger(p[1]);
   string sym       = p[2];
   bool   isBuy     = (p[3] == "BUY");
   double entry     = StringToDouble(p[4]);
   double slAbs     = StringToDouble(p[5]);
   double tpAbs     = StringToDouble(p[6]);
   double slDist    = StringToDouble(p[7]);
   double tpDist    = StringToDouble(p[8]);
   double lotsAsked = StringToDouble(p[9]);
   double riskPct   = StringToDouble(p[10]);
   long   expires   = StringToInteger(p[11]);
   long   magic     = (long)CfgNum("magic", 770011);
   string doneKey   = "GSB_DONE_" + IntegerToString(id);

   if(GlobalVariableCheck(doneKey))
      return true;                                   // already handled in an earlier poll

   ulong existing = FindPositionByTag(id, magic);
   if(existing > 0)                                   // it is already open (e.g. the report got lost): just tell the app
   {
      if(PositionSelectByTicket(existing))
         Report(id, "open", existing, PositionGetDouble(POSITION_PRICE_OPEN), PositionGetDouble(POSITION_VOLUME), 0, "");
      TrackAdd(id, existing);
      GlobalVariableSet(doneKey, (double)TimeCurrent());
      return true;
   }

   if(expires > 0 && (long)TimeGMT() > expires)
   {
      Report(id, "skip", 0, 0, 0, 0, "Expired before the EA could place it");
      GlobalVariableSet(doneKey, (double)TimeCurrent());
      return true;
   }

   // ---- safety locks
   bool demoOnlyApp = (CfgGet("demo_only", "1") == "1");
   if(IsRealAccount() && (!InpAllowLiveAccount || demoOnlyApp))
   {
      Report(id, "skip", 0, 0, 0, 0, "REAL account blocked. To allow it: tick InpAllowLiveAccount in the EA inputs AND untick 'Demo only' in the app.");
      GlobalVariableSet(doneKey, (double)TimeCurrent());
      return true;
   }
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) || !AccountInfoInteger(ACCOUNT_TRADE_ALLOWED))
   {
      g_status = "trading is not allowed: press the Algo Trading button";
      return false;                                  // retry later (until the order expires)
   }

   if(!SymbolSelect(sym, true))
   {
      Report(id, "reject", 0, 0, 0, 0, "Symbol " + sym + " not found at your broker. Set the broker suffix or a name override in Expert options.");
      GlobalVariableSet(doneKey, (double)TimeCurrent());
      return true;
   }
   if((ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(sym, SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_DISABLED)
   {
      Report(id, "reject", 0, 0, 0, 0, "Trading is disabled for " + sym + " at your broker");
      GlobalVariableSet(doneKey, (double)TimeCurrent());
      return true;
   }

   int    digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   double point  = SymbolInfoDouble(sym, SYMBOL_POINT);
   double ask    = SymbolInfoDouble(sym, SYMBOL_ASK);
   double bid    = SymbolInfoDouble(sym, SYMBOL_BID);
   if(ask <= 0 || bid <= 0 || point <= 0)
      return false;                                  // no quote yet (market closed?) - try again

   // ---- spread filter (retry: spreads spike around rollover and news)
   double maxSpread = CfgNum("max_spread", 0);
   long   spread    = SymbolInfoInteger(sym, SYMBOL_SPREAD);
   if(maxSpread > 0 && spread > (long)maxSpread)
   {
      g_status = sym + ": spread " + IntegerToString(spread) + " > limit " + DoubleToString(maxSpread, 0) + ", waiting";
      return false;
   }

   // ---- price and levels
   double price = isBuy ? ask : bid;
   double sl, tp;
   string priceMode = CfgGet("price_mode", "relative");
   if(priceMode == "absolute")
   {
      double driftR = CfgNum("max_drift_r", 0.5);
      if(slDist > 0 && MathAbs(price - entry) > driftR * slDist)
      {
         Report(id, "skip", 0, 0, 0, 0, "Price moved " + DoubleToString(MathAbs(price - entry) / slDist, 2) + "R from the signal entry (limit " + DoubleToString(driftR, 2) + "R)");
         GlobalVariableSet(doneKey, (double)TimeCurrent());
         return true;
      }
      sl = slAbs;
      tp = tpAbs;
   }
   else
   {
      sl = isBuy ? price - slDist : price + slDist;
      tp = isBuy ? price + tpDist : price - tpDist;
   }
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);
   if((isBuy && (sl >= price || tp <= price)) || (!isBuy && (sl <= price || tp >= price)))
   {
      Report(id, "reject", 0, 0, 0, 0, "Stop / target are on the wrong side of the live price");
      GlobalVariableSet(doneKey, (double)TimeCurrent());
      return true;
   }
   double stopsLevel = (double)SymbolInfoInteger(sym, SYMBOL_TRADE_STOPS_LEVEL) * point;
   if(MathAbs(price - sl) < stopsLevel || MathAbs(tp - price) < stopsLevel)
   {
      Report(id, "reject", 0, 0, 0, 0, "Stop or target closer than the broker's minimum distance");
      GlobalVariableSet(doneKey, (double)TimeCurrent());
      return true;
   }

   // ---- lots
   double lots;
   double maxLots = CfgNum("max_lots", 1.0);
   if(CfgGet("lot_mode", "risk") == "risk" && riskPct > 0)
   {
      double budget = AccountInfoDouble(ACCOUNT_BALANCE) * riskPct / 100.0;
      double lpl    = LossPerLot(sym, isBuy, price, sl);
      if(lpl <= 0)
      {
         Report(id, "reject", 0, 0, 0, 0, "Cannot size the trade by risk for " + sym + " (no tick value). Use fixed lots in Expert options.");
         GlobalVariableSet(doneKey, (double)TimeCurrent());
         return true;
      }
      lots = budget / lpl;
      double vmin = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
      if(lots < vmin && vmin * lpl > budget * 1.5)
      {
         Report(id, "skip", 0, 0, 0, 0, "Smallest lot (" + DoubleToString(vmin, 2) + ") would risk more than 1.5x your " + DoubleToString(riskPct, 2) + "% budget. Raise the risk % or use a bigger account.");
         GlobalVariableSet(doneKey, (double)TimeCurrent());
         return true;
      }
   }
   else
      lots = lotsAsked > 0 ? lotsAsked : CfgNum("lots", 0.01);
   if(lots > maxLots)
      lots = maxLots;
   lots = NormalizeLots(sym, lots);

   // ---- margin
   double margin = 0.0;
   if(OrderCalcMargin(isBuy ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, sym, lots, price, margin))
   {
      if(margin > AccountInfoDouble(ACCOUNT_MARGIN_FREE) * 0.9)
      {
         Report(id, "reject", 0, 0, 0, 0, "Not enough free margin for " + DoubleToString(lots, 2) + " lots");
         GlobalVariableSet(doneKey, (double)TimeCurrent());
         return true;
      }
   }

   // ---- place it
   string comment = Tag(id) + " " + CfgGet("comment", "GoldSignal");
   if(StringLen(comment) > 31)
      comment = StringSubstr(comment, 0, 31);
   g_trade.SetExpertMagicNumber((ulong)magic);
   g_trade.SetDeviationInPoints((ulong)CfgNum("slippage", 30));
   g_trade.SetTypeFillingBySymbol(sym);
   bool ok = isBuy ? g_trade.Buy(lots, sym, 0.0, sl, tp, comment) : g_trade.Sell(lots, sym, 0.0, sl, tp, comment);
   uint rc = g_trade.ResultRetcode();
   if(ok && (rc == TRADE_RETCODE_DONE || rc == TRADE_RETCODE_PLACED || rc == TRADE_RETCODE_DONE_PARTIAL))
   {
      GlobalVariableSet(doneKey, (double)TimeCurrent());
      ulong tk = FindPositionByTag(id, magic);
      if(tk == 0)
         tk = g_trade.ResultOrder();
      TrackAdd(id, tk);
      Report(id, "open", tk, g_trade.ResultPrice(), g_trade.ResultVolume(), 0, "");
      g_status = "opened " + sym + " " + p[3] + " " + DoubleToString(lots, 2) + " lots";
      return true;
   }
   // transient broker answers: try again on the next poll instead of giving up
   if(rc == TRADE_RETCODE_REQUOTE || rc == TRADE_RETCODE_PRICE_CHANGED || rc == TRADE_RETCODE_PRICE_OFF ||
      rc == TRADE_RETCODE_TIMEOUT || rc == TRADE_RETCODE_CONNECTION || rc == TRADE_RETCODE_TOO_MANY_REQUESTS ||
      rc == TRADE_RETCODE_MARKET_CLOSED)
   {
      g_status = sym + ": " + g_trade.ResultRetcodeDescription() + " - will retry";
      return false;
   }
   Report(id, "reject", 0, 0, 0, 0, g_trade.ResultRetcodeDescription() + " (code " + IntegerToString((long)rc) + ")");
   GlobalVariableSet(doneKey, (double)TimeCurrent());
   return true;
}

void ExecuteClose(string &p[])
{
   // CLOSE|id|ticket
   if(ArraySize(p) < 2)
      return;
   long id = StringToInteger(p[1]);
   long magic = (long)CfgNum("magic", 770011);
   ulong tk = FindPositionByTag(id, magic);
   if(tk == 0)
      return;
   g_trade.SetExpertMagicNumber((ulong)magic);
   g_trade.PositionClose(tk);
}

//+------------------------------------------------------------------+
//| break-even (done here, on live prices, not on the app's candles)  |
//+------------------------------------------------------------------+
void ManageBreakEven()
{
   double beR = CfgNum("be_at_r", 1.0);
   if(beR <= 0)
      return;
   long magic = (long)CfgNum("magic", 770011);
   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != magic)
         continue;
      if(StringFind(PositionGetString(POSITION_COMMENT), "GS#") != 0)
         continue;
      string sym    = PositionGetString(POSITION_SYMBOL);
      double openP  = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl     = PositionGetDouble(POSITION_SL);
      double tp     = PositionGetDouble(POSITION_TP);
      double point  = SymbolInfoDouble(sym, SYMBOL_POINT);
      int    digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
      bool   isBuy  = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
      double stopsLevel = (double)SymbolInfoInteger(sym, SYMBOL_TRADE_STOPS_LEVEL) * point;
      if(sl <= 0 || point <= 0)
         continue;
      if(isBuy)
      {
         double risk = openP - sl;
         if(risk <= point)                           // stop is already at / past the entry: break-even is done
            continue;
         double bid = SymbolInfoDouble(sym, SYMBOL_BID);
         if(bid - openP >= beR * risk && bid - openP > stopsLevel)
            g_trade.PositionModify(tk, NormalizeDouble(openP, digits), tp);
      }
      else
      {
         double risk = sl - openP;
         if(risk <= point)
            continue;
         double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
         if(openP - ask >= beR * risk && openP - ask > stopsLevel)
            g_trade.PositionModify(tk, NormalizeDouble(openP, digits), tp);
      }
   }
}

//+------------------------------------------------------------------+
//| tell the app when a position we opened has closed                 |
//+------------------------------------------------------------------+
void CheckClosed()
{
   for(int i = ArraySize(g_trkId) - 1; i >= 0; i--)
   {
      ulong tk = g_trkTicket[i];
      if(tk > 0 && PositionSelectByTicket(tk))
         continue;                                   // still open
      double profit = 0.0;
      double price  = 0.0;
      if(tk > 0 && HistorySelectByPosition(tk))
      {
         int dn = HistoryDealsTotal();
         for(int d = 0; d < dn; d++)
         {
            ulong dt = HistoryDealGetTicket(d);
            if(dt == 0)
               continue;
            profit += HistoryDealGetDouble(dt, DEAL_PROFIT) + HistoryDealGetDouble(dt, DEAL_SWAP) + HistoryDealGetDouble(dt, DEAL_COMMISSION);
            long entryType = HistoryDealGetInteger(dt, DEAL_ENTRY);
            if(entryType == DEAL_ENTRY_OUT || entryType == DEAL_ENTRY_OUT_BY || entryType == DEAL_ENTRY_INOUT)
               price = HistoryDealGetDouble(dt, DEAL_PRICE);
         }
      }
      else if(tk > 0)
         continue;                                   // history not loaded yet: look again next tick
      string resp;
      string b = "{\"id\":" + IntegerToString(g_trkId[i]) + ",\"event\":\"closed\",\"ticket\":\"" + IntegerToString((long)tk) + "\"";
      if(price > 0)
         b += ",\"price\":" + DoubleToString(price, 8);
      b += ",\"profit\":" + DoubleToString(profit, 2) + "}";
      int code = Post("/bridge/ea/report", b, resp);
      if(code == 200)
         TrackRemoveAt(i);
   }
}

//+------------------------------------------------------------------+
//| poll                                                              |
//+------------------------------------------------------------------+
void ParseCfg(string line)
{
   string parts[];
   int n = StringSplit(line, '|', parts);
   ArrayResize(g_cfgKeys, 0);
   ArrayResize(g_cfgVals, 0);
   for(int i = 1; i < n; i++)
   {
      int eq = StringFind(parts[i], "=");
      if(eq <= 0)
         continue;
      int k = ArraySize(g_cfgKeys);
      ArrayResize(g_cfgKeys, k + 1);
      ArrayResize(g_cfgVals, k + 1);
      g_cfgKeys[k] = StringSubstr(parts[i], 0, eq);
      g_cfgVals[k] = StringSubstr(parts[i], eq + 1);
   }
}

void BuildHistoryQueue()
{
   ArrayResize(g_hSym, 0);
   ArrayResize(g_hTf, 0);
   ArrayResize(g_hTfName, 0);
   string list = CfgGet("pairs", "");
   if(list == "")
      return;
   string syms[];
   int n = StringSplit(list, ',', syms);
   ENUM_TIMEFRAMES tfs[5] = {PERIOD_M5, PERIOD_M15, PERIOD_M30, PERIOD_H1, PERIOD_D1};
   string names[5] = {"5m", "15m", "30m", "1h", "1d"};
   for(int i = 0; i < n; i++)
   {
      if(syms[i] == "")
         continue;
      if(!SymbolSelect(syms[i], true))
         continue;
      for(int t = 0; t < 5; t++)
      {
         int k = ArraySize(g_hSym);
         ArrayResize(g_hSym, k + 1);
         ArrayResize(g_hTf, k + 1);
         ArrayResize(g_hTfName, k + 1);
         g_hSym[k] = syms[i];
         g_hTf[k] = tfs[t];
         g_hTfName[k] = names[t];
      }
   }
   g_hIdx = 0;
   g_hOff = 0;
   g_hWant = 0;
   g_hFails = 0;
   g_historyBuilt = true;
}

void Poll()
{
   string body = "{\"version\":\"" + GS_VERSION + "\",\"account\":" + AccountJson() + "}";
   string resp;
   int code = Post("/bridge/ea/poll", body, resp);
   if(code == 401)
   {
      g_status = "the app rejected the token - download a fresh EA from the Expert card";
      return;
   }
   if(code != 200)
   {
      if(code != -1)
         g_status = "server answered HTTP " + IntegerToString(code);
      return;
   }
   string lines[];
   int n = StringSplit(resp, '\n', lines);
   string openLines[];
   string closeLines[];
   ArrayResize(openLines, 0);
   ArrayResize(closeLines, 0);
   for(int i = 0; i < n; i++)
   {
      string ln = lines[i];
      StringReplace(ln, "\r", "");
      if(StringLen(ln) < 3)
         continue;
      if(StringFind(ln, "CFG|") == 0)
         ParseCfg(ln);
      else if(StringFind(ln, "OPEN|") == 0)
      {
         int k = ArraySize(openLines);
         ArrayResize(openLines, k + 1);
         openLines[k] = ln;
      }
      else if(StringFind(ln, "CLOSE|") == 0)
      {
         int k = ArraySize(closeLines);
         ArrayResize(closeLines, k + 1);
         closeLines[k] = ln;
      }
   }
   g_status = "connected, mode " + CfgGet("mode", "?");
   if(InpUploadHistory && !g_historyBuilt && ArraySize(g_cfgKeys) > 0)
      BuildHistoryQueue();
   for(int i = 0; i < ArraySize(openLines); i++)
   {
      string p[];
      StringSplit(openLines[i], '|', p);
      ExecuteOpen(p);
   }
   for(int i = 0; i < ArraySize(closeLines); i++)
   {
      string p[];
      StringSplit(closeLines[i], '|', p);
      ExecuteClose(p);
   }
}

//+------------------------------------------------------------------+
//| candle history -> the app (brain / replay / backtest)             |
//+------------------------------------------------------------------+
void UploadStep()
{
   if(!InpUploadHistory || !g_historyBuilt || g_hIdx >= ArraySize(g_hSym))
      return;
   string sym = g_hSym[g_hIdx];
   ENUM_TIMEFRAMES tf = g_hTf[g_hIdx];
   string stamp = "GSB_HIST_" + sym + "_" + g_hTfName[g_hIdx];
   if(g_hOff == 0)
   {
      int have = (int)SeriesInfoInteger(sym, tf, SERIES_BARS_COUNT);
      // first time: everything we can; later starts: only the most recent bars (fills any gap)
      int cap = GlobalVariableCheck(stamp) ? 3000 : InpHistoryBars;
      g_hWant = MathMin(have - 2, cap);
      if(g_hWant < 50)
      {
         g_hFails++;
         if(g_hFails > 3)                            // the terminal has not loaded this series: skip it
         {
            g_hIdx++;
            g_hFails = 0;
         }
         return;
      }
   }
   int chunk = 1000;
   if(g_hOff + chunk > g_hWant)
      chunk = g_hWant - g_hOff;
   if(chunk <= 0)
   {
      GlobalVariableSet(stamp, (double)TimeCurrent());
      g_hIdx++;
      g_hOff = 0;
      return;
   }
   MqlRates rates[];
   int got = CopyRates(sym, tf, 1 + g_hOff, chunk, rates);     // start at 1: the forming candle is never sent
   if(got <= 0)
   {
      g_hFails++;
      if(g_hFails > 3)
      {
         g_hIdx++;
         g_hOff = 0;
         g_hFails = 0;
      }
      return;
   }
   g_hFails = 0;
   int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   double off = (double)((long)TimeTradeServer() - (long)TimeGMT()) / 3600.0;
   off = MathRound(off * 2.0) / 2.0;                           // broker server time minus UTC, in hours
   string b = "{\"symbol\":\"" + JsonEsc(sym) + "\",\"tf\":\"" + g_hTfName[g_hIdx] + "\",\"tz_offset_hours\":" + DoubleToString(off, 1) + ",\"rows\":[";
   for(int i = 0; i < got; i++)
   {
      if(i > 0)
         b += ",";
      b += "[" + IntegerToString((long)rates[i].time) + "," + DoubleToString(rates[i].open, digits) + "," + DoubleToString(rates[i].high, digits) + "," +
           DoubleToString(rates[i].low, digits) + "," + DoubleToString(rates[i].close, digits) + "," + IntegerToString((long)rates[i].tick_volume) + "]";
   }
   b += "]}";
   string resp;
   int code = Post("/bridge/ea/candles", b, resp);
   if(code == 200)
   {
      g_hOff += got;
      g_status = "sending history " + sym + " " + g_hTfName[g_hIdx] + " " + IntegerToString(g_hOff) + "/" + IntegerToString(g_hWant);
      if(g_hOff >= g_hWant)
      {
         GlobalVariableSet(stamp, (double)TimeCurrent());
         g_hIdx++;
         g_hOff = 0;
      }
   }
   else
   {
      g_hFails++;
      if(code == 400 || g_hFails > 3)                          // the app refused this pair (not in allowed pairs) or keeps failing
      {
         g_hIdx++;
         g_hOff = 0;
         g_hFails = 0;
      }
   }
}

//+------------------------------------------------------------------+
void ShowStatus()
{
   string s = "GoldSignal Bridge " + GS_VERSION + "\n" + g_status;
   s += "\nAccount " + IntegerToString((long)AccountInfoInteger(ACCOUNT_LOGIN)) + (IsRealAccount() ? " (REAL)" : " (demo)");
   if(IsRealAccount() && !InpAllowLiveAccount)
      s += "\nReal account: trading blocked (InpAllowLiveAccount is off)";
   Comment(s);
}

int OnInit()
{
   if(StringFind(InpToken, "gsb_") != 0)
   {
      Alert("GoldSignalBridge: the token is missing. Download the EA from the app (Profile > Expert) - it contains your token.");
      return INIT_PARAMETERS_INCORRECT;
   }
   if(StringFind(InpApiUrl, "http") != 0)
   {
      Alert("GoldSignalBridge: InpApiUrl is not set.");
      return INIT_PARAMETERS_INCORRECT;
   }
   TrackReload();
   EventSetTimer(1);
   g_status = "starting";
   ShowStatus();
   Print("GoldSignalBridge ", GS_VERSION, " started. Server ", ServerBase(), ". Remember: add that address under Tools > Options > Expert Advisors > Allow WebRequest.");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   Comment("");
}

void OnTimer()
{
   ulong now = GetTickCount64();
   if(now - g_lastPollMs >= 3000)
   {
      g_lastPollMs = now;
      Poll();
   }
   ManageBreakEven();
   CheckClosed();
   if(now - g_lastPollMs >= 900)                        // history goes out between polls, one chunk at a time
      UploadStep();
   ShowStatus();
}

void OnTick()
{
}
