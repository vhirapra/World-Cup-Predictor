"""
World Cup 2026 Predictor Dashboard
Interactive visualizations of tournament predictions
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dashboard_helpers import (
    load_models, get_group_stage_matches, get_group_standings,
    get_probable_matches, run_tournament_simulation, get_flag_emoji,
    get_sample_bracket, GROUPS, Match
)

st.set_page_config(page_title="2026 World Cup Predictor", layout="wide", initial_sidebar_state="expanded")

st.title("⚽ 2026 FIFA World Cup Predictor")
st.markdown("**Interactive Dashboard** - Match predictions, tournament bracket, and win probabilities")

# Load models (cached)
@st.cache_resource
def cached_models():
    poisson_model, xgboost_model, encoder_dict = load_models()
    return poisson_model, xgboost_model, encoder_dict

try:
    poisson_model, xgboost_model, encoder_dict = cached_models()
except FileNotFoundError:
    st.error("❌ Models not found! Please run `python main.py` first to train and save models.")
    st.stop()

# Sidebar controls
st.sidebar.header("⚙️ Controls")
num_simulations = st.sidebar.slider("Tournament Simulations", min_value=50, max_value=1000, value=100, step=50)

if st.sidebar.button("🔄 Run Tournament Simulation", use_container_width=True):
    st.session_state.run_simulation = True
else:
    st.session_state.run_simulation = False

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Group Stage", "🏆 Bracket", "📈 Closest Matchups", "🎯 Statistics"])

# ================== TAB 1: GROUP STAGE ==================
with tab1:
    st.header("Group Stage Predictions")
    st.markdown("All 72 group stage matchups with win/draw/loss probabilities")

    # Get group stage matches
    matches = get_group_stage_matches(GROUPS, poisson_model, xgboost_model, encoder_dict)

    # Get standings
    standings = get_group_standings(GROUPS, poisson_model, xgboost_model, encoder_dict)

    for group_name in sorted(standings.keys()):
        with st.expander(f"**{group_name}**", expanded=False):
            # Matchups
            st.subheader("Matches")
            group_teams = GROUPS[group_name]
            group_matches = [m for m in matches if m.stage.endswith(f"({group_name})")]

            cols = st.columns([2, 1, 1, 1, 1, 1, 1])
            with cols[0]:
                st.write("**Matchup**")
            with cols[1]:
                st.write("**Win %**")
            with cols[2]:
                st.write("**Draw %**")
            with cols[3]:
                st.write("**Loss %**")
            with cols[4]:
                st.write("**xG A**")
            with cols[5]:
                st.write("**xG B**")
            with cols[6]:
                st.write("**Winner**")

            for match in group_matches:
                cols = st.columns([2, 1, 1, 1, 1, 1, 1])
                with cols[0]:
                    flag_a = get_flag_emoji(match.team_a)
                    flag_b = get_flag_emoji(match.team_b)
                    st.write(f"{flag_a} {match.team_a} vs {flag_b} {match.team_b}")
                with cols[1]:
                    st.write(f"{match.p_a*100:.1f}%")
                with cols[2]:
                    st.write(f"{match.p_draw*100:.1f}%")
                with cols[3]:
                    st.write(f"{match.p_b*100:.1f}%")
                with cols[4]:
                    st.write(f"{match.xg_a:.2f}")
                with cols[5]:
                    st.write(f"{match.xg_b:.2f}")
                with cols[6]:
                    winner_flag = get_flag_emoji(match.predicted_winner())
                    st.write(f"**{winner_flag}**")

            # Standings
            st.subheader("Predicted Standings")
            standings_data = []
            for entry in standings[group_name]:
                standings_data.append({
                    "Pos": entry['rank'],
                    "Team": f"{get_flag_emoji(entry['team'])} {entry['team']}",
                    "Pts": int(entry['pts']) if entry['pts'] == int(entry['pts']) else f"{entry['pts']:.1f}",
                    "GF": int(entry['gf']) if entry['gf'] == int(entry['gf']) else f"{entry['gf']:.1f}",
                    "GA": int(entry['ga']),
                    "Advances": "✅" if entry['advances'] else "❌"
                })

            st.dataframe(
                pd.DataFrame(standings_data),
                hide_index=True,
                use_container_width=True
            )

# ================== TAB 2: KNOCKOUT BRACKET ==================
with tab2:
    st.header("Tournament Bracket")
    st.markdown("*Visual tournament progression with predicted winners at each stage*")

    if st.session_state.run_simulation:
        with st.spinner(f"🎲 Running {num_simulations} tournament simulations..."):
            st.session_state.tournament_results = run_tournament_simulation(poisson_model, xgboost_model, encoder_dict, num_simulations)

    if 'tournament_results' in st.session_state:
        results = st.session_state.tournament_results
        st.success(f"✅ Tournament simulation complete")

        # Display champion
        champion = list(results.keys())[0]
        champion_prob = results[champion]
        champion_flag = get_flag_emoji(champion)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                label="🏆 Most Likely Champion",
                value=f"{champion_flag} {champion}",
                delta=f"{champion_prob:.1f}% probability"
            )

        # Top contenders
        st.subheader("Top 10 Contenders")
        top_10 = list(results.items())[:10]
        contenders_df = pd.DataFrame([
            {"Rank": i+1, "Team": f"{get_flag_emoji(team)} {team}", "Probability": f"{prob:.1f}%"}
            for i, (team, prob) in enumerate(top_10)
        ])
        st.dataframe(contenders_df, hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("🌳 Example Simulated Bracket Path")
        st.caption("This shows one potential deterministic path based on the simulation data.")

        bracket_data = get_sample_bracket(poisson_model, xgboost_model, encoder_dict)

        for stage_name, matches in bracket_data.items():
            st.markdown(f"#### {stage_name}")
            cols = st.columns(min(len(matches), 4))
            for i, match in enumerate(matches):
                with cols[i % 4]:
                    team_a_disp = f"**{match['team_a']}**" if match['winner'] == match['team_a'] else match['team_a']
                    team_b_disp = f"**{match['team_b']}**" if match['winner'] == match['team_b'] else match['team_b']
                    st.info(f"{get_flag_emoji(match['team_a'])} {team_a_disp}\n\n"
                            f"{get_flag_emoji(match['team_b'])} {team_b_disp}\n\n"
                            f"*{match['score']}*")

    else:
        st.info("👈 Click 'Run Tournament Simulation' in the sidebar to generate bracket predictions")

# ================== TAB 3: CLOSEST MATCHUPS ==================
with tab3:
    st.header("Most Competitive Matchups")
    st.markdown("*Matches ranked by closeness of predicted outcome (closest = 50-50 split)*")

    matches = get_group_stage_matches(GROUPS, poisson_model, xgboost_model, encoder_dict)
    probable = get_probable_matches(matches, top_n=20)

    competitive_data = []
    for match in probable:
        competitive_data.append({
            "Matchup": f"{get_flag_emoji(match.team_a)} {match.team_a} vs {get_flag_emoji(match.team_b)} {match.team_b}",
            "Stage": match.stage.split("(")[0].strip(),
            "Team A Win": f"{match.p_a*100:.1f}%",
            "Draw": f"{match.p_draw*100:.1f}%",
            "Team B Win": f"{match.p_b*100:.1f}%",
            "Competitiveness": f"{(1-match.competitiveness)*100:.0f}%"
        })

    st.dataframe(
        pd.DataFrame(competitive_data),
        hide_index=True,
        use_container_width=True
    )

    # Visualization
    st.subheader("Probability Distribution of Top Matchups")
    fig_data = []
    for i, match in enumerate(probable[:10]):
        fig_data.append({
            "Matchup": f"{match.team_a} vs {match.team_b}",
            "Team A Win": match.p_a * 100,
            "Draw": match.p_draw * 100,
            "Team B Win": match.p_b * 100,
        })

    fig = go.Figure()
    for match in fig_data:
        matchup = match["Matchup"]
        fig.add_trace(go.Bar(
            name=matchup,
            x=["Team A Win", "Draw", "Team B Win"],
            y=[match["Team A Win"], match["Draw"], match["Team B Win"]],
            text=[f"{v:.1f}%" for v in [match["Team A Win"], match["Draw"], match["Team B Win"]]],
            textposition="auto",
        ))

    fig.update_layout(
        barmode="group",
        height=500,
        xaxis_title="Outcome",
        yaxis_title="Probability (%)",
        showlegend=True,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# ================== TAB 4: STATISTICS ==================
with tab4:
    st.header("Tournament Statistics & Insights")

    matches = get_group_stage_matches(GROUPS, poisson_model, xgboost_model, encoder_dict)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_matches = len(matches)
        st.metric("Total Group Matches", total_matches)
    with col2:
        avg_draw_prob = np.mean([m.p_draw for m in matches])
        st.metric("Avg Draw Probability", f"{avg_draw_prob*100:.1f}%")
    with col3:
        predicted_upsets = sum(1 for m in matches if abs(m.p_a - m.p_b) > 0.3)
        st.metric("Strong Favorites", predicted_upsets)
    with col4:
        balanced_matches = sum(1 for m in matches if abs(m.p_a - m.p_b) < 0.1)
        st.metric("Closely Balanced", balanced_matches)

    # Probability distribution histogram
    st.subheader("Match Outcome Probability Distribution")
    prob_diffs = [abs(m.p_a - m.p_b) for m in matches]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=prob_diffs,
        nbinsx=20,
        marker_color='rgba(55, 128, 191, 0.7)',
        name="Probability Difference"
    ))
    fig.update_layout(
        title="Distribution of Outcome Predictability",
        xaxis_title="Win Probability Gap (|Team A - Team B|)",
        yaxis_title="Number of Matches",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interesting facts
    st.subheader("🔍 Interesting Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Most Likely Draws**")
        most_draws = sorted(matches, key=lambda m: m.p_draw, reverse=True)[:3]
        for match in most_draws:
            flag_a = get_flag_emoji(match.team_a)
            flag_b = get_flag_emoji(match.team_b)
            st.caption(f"{flag_a} {match.team_a} vs {flag_b} {match.team_b}: {match.p_draw*100:.1f}%")

    with col2:
        st.write("**Biggest Upsets (Underdog Win %)**")
        upsets = []
        for match in matches:
            underdog_win = min(match.p_a, match.p_b)
            upsets.append((match, underdog_win))
        upsets = sorted(upsets, key=lambda x: x[1], reverse=True)[:3]
        for match, prob in upsets:
            flag_a = get_flag_emoji(match.team_a)
            flag_b = get_flag_emoji(match.team_b)
            underdog = match.team_a if match.p_a < match.p_b else match.team_b
            flag_underdog = get_flag_emoji(underdog)
            st.caption(f"{flag_underdog} {underdog} upset: {prob*100:.1f}%")

# Footer
st.divider()
st.markdown("""
**About this Dashboard:**
- Powered by Poisson Regression + XGBoost Residual Modeling
- Predictions based on international football history (1990-2024)
- Group stage: 72 matches with probability distributions
- Bracket: Monte Carlo tournament simulations
- All probabilities are normalized (win + draw + loss = 100%)
""")
st.markdown("Created with ❤️ using Streamlit | Data: Kaggle International Football Results")
