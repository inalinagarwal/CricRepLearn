const IND_BAT =
  "Rohit Sharma,Shubman Gill,Virat Kohli,Ishan Kishan,Shreyas Iyer,Washington Sundar,Shivam Dube,Axar Patel,Gurnoor Brar,Jasprit Bumrah,Prasidh Krishna";
const ENG_BAT =
  "Jacob Bethell,Ben Duckett,Joe Root,Harry Brook,Jos Buttler,Sam Curran,Will Jacks,Gus Atkinson,Jofra Archer,Adil Rashid,Saqib Mahmood";
const IND_BOWL = "Jasprit Bumrah,Prasidh Krishna,Gurnoor Brar,Axar Patel,Washington Sundar";
const ENG_BOWL = "Jofra Archer,Gus Atkinson,Saqib Mahmood,Adil Rashid,Sam Curran";

function fillDefaults() {
  const dream = document.getElementById("form-dream");
  dream.team_a_batters.value = IND_BAT;
  dream.team_b_batters.value = ENG_BAT;
  dream.team_a_bowlers.value = IND_BOWL;
  dream.team_b_bowlers.value = ENG_BOWL;

  const match = document.getElementById("form-match");
  match.first_batters.value = IND_BAT;
  match.chase_batters.value = ENG_BAT;
  match.first_bowlers.value = ENG_BOWL;
  match.chase_bowlers.value = IND_BOWL;
}

function switchMode(mode) {
  document.querySelectorAll(".mode").forEach((btn) => {
    btn.setAttribute("aria-pressed", String(btn.dataset.mode === mode));
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${mode}`);
  });
}

function formToObject(form) {
  const data = {};
  new FormData(form).forEach((value, key) => {
    if (typeof value === "string" && value.trim() === "") {
      data[key] = null;
    } else if (key === "sims" || key === "max_balls") {
      data[key] = Number(value);
    } else {
      data[key] = value;
    }
  });
  return data;
}

function formatDream(result) {
  const xi = result.best_xi;
  const lines = [
    `Venue ${result.venue || "—"}  date ${result.match_date || "—"}  sims ${result.n_sims}`,
    `Toss A first ${result.toss_a.first_expected_runs?.toFixed(1)} → chase ${result.toss_a.chase_expected_runs?.toFixed(1)}  P(chase)=${result.toss_a.p_chase_win?.toFixed(2)}`,
    `Toss B first ${result.toss_b.first_expected_runs?.toFixed(1)} → chase ${result.toss_b.chase_expected_runs?.toFixed(1)}  P(chase)=${result.toss_b.p_chase_win?.toFixed(2)}`,
    "",
    `BEST XI  obj=${xi.objective_score.toFixed(1)}  credits=${xi.credits_used ?? "—"}`,
    `C  ${xi.captain.player_name}`,
    `VC ${xi.vice_captain.player_name}`,
    "",
  ];
  for (const p of xi.players) {
    lines.push(
      `${p.role.padEnd(4)} ${p.team.padEnd(3)} ${p.player_name.padEnd(22)} ${p.fantasy_points.toFixed(1)}`
    );
  }
  return lines.join("\n");
}

function formatMatch(result) {
  const m = result.match;
  const lines = [
    `Venue ${result.venue || "—"}  sims ${result.n_sims}`,
    `First ${m.first_expected_runs?.toFixed(1)}  Chase ${m.chase_expected_runs?.toFixed(1)}  P(chase)=${m.p_chase_win?.toFixed(2)}`,
  ];
  for (const [label, block] of [
    ["FIRST", result.first_innings],
    ["CHASE", result.chase_innings],
  ]) {
    lines.push("", `=== ${label} ===`);
    for (const b of block.batters || []) {
      lines.push(
        `${b.player_name.padEnd(22)} runs ${Number(b.expected_runs || 0).toFixed(1)}  balls ${Number(b.expected_balls || 0).toFixed(1)}`
      );
    }
    lines.push("-- bowlers --");
    for (const b of block.bowlers || []) {
      lines.push(
        `${b.player_name.padEnd(22)} wkts ${Number(b.expected_wickets || 0).toFixed(2)}  overs ${Number(b.expected_overs || 0).toFixed(2)}`
      );
    }
    if (block.overs?.length) {
      lines.push(
        "overs: " +
          block.overs
            .map(
              (o) =>
                `${Number(o.over) + 1}:${Number(o.expected_runs || 0).toFixed(1)}/${Number(o.expected_wickets || 0).toFixed(2)}`
            )
            .join("  ")
      );
    }
  }
  return lines.join("\n");
}

function formatDive(result) {
  const lines = [
    `Batter ${result.batter?.player_name || result.batter}`,
    `Venue ${result.venue || "—"}`,
    `Expected runs ${result.expected_runs}`,
    `Expected balls ${result.expected_balls}`,
  ];
  if (result.warning) lines.push(`Warning: ${result.warning}`);
  const rows = result.attack || [];
  for (const row of rows) {
    const name = row.bowler_name || row.player_name || "?";
    lines.push(
      `${String(name).padEnd(22)} runs ${Number(row.expected_runs || 0).toFixed(2)}  balls ${Number(row.expected_balls_faced || 0).toFixed(2)}  ${row.level || ""}`
    );
  }
  return lines.join("\n");
}

async function run(endpoint, body, formatter, title) {
  const results = document.getElementById("results");
  const output = document.getElementById("output");
  const status = document.getElementById("status");
  const titleEl = document.getElementById("results-title");
  results.hidden = false;
  titleEl.textContent = title;
  status.textContent = "Running Monte Carlo…";
  output.textContent = "";
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    status.textContent = "Done";
    output.textContent = formatter(data);
  } catch (err) {
    status.textContent = "Failed";
    output.textContent = String(err.message || err);
  }
}

document.querySelectorAll(".mode").forEach((btn) => {
  btn.addEventListener("click", () => switchMode(btn.dataset.mode));
});

document.getElementById("form-dream").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector("button[type=submit]");
  btn.disabled = true;
  await run("/api/dream-xi", formToObject(e.target), formatDream, "Dream XI");
  btn.disabled = false;
});

document.getElementById("form-match").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector("button[type=submit]");
  btn.disabled = true;
  await run("/api/match-sim", formToObject(e.target), formatMatch, "Match sim");
  btn.disabled = false;
});

document.getElementById("form-dive").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = e.target.querySelector("button[type=submit]");
  btn.disabled = true;
  await run("/api/player-dive", formToObject(e.target), formatDive, "Player dive");
  btn.disabled = false;
});

fillDefaults();
