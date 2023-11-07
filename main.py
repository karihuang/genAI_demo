import streamlit as st
import langchain_helper as lch
import textwrap

st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label="YouTube video URL?",
            max_chars=100
            )
        query = st.sidebar.text_area(
            label="Ask me about the video?",
            max_chars=100,
            key="query"
            )
        submit = st.form_submit_button(label='Submit')
        
st.markdown("""
            <style>
                div[data-testid="column"] {
                    width: fit-content !important;
                    flex: unset;
                }
                div[data-testid="column"] * {
                    width: fit-content !important;
                }
            </style>
            """, unsafe_allow_html=True)
col1, col2 = st.columns([1,1])

with col1:
    summarize_button = st.button('Summarize')
with col2:
    name_generate_button = st.button('Generate Title')
    
db = None
if youtube_url:
    db = lch.create_db_from_youtube_video_url(youtube_url)

if submit and query and db:
    response = lch.get_response_from_query(db, query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=75))

if name_generate_button and not submit and db:
    response = lch.generate_titles(db)
    st.subheader("Three Recommended Titles:")
    st.text(response)

if summarize_button and not submit and db:
    response = lch.summerize_text(db)
    st.subheader("Summary:")
    st.text(textwrap.fill(response, width=75))