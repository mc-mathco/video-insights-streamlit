#----------------------------------------------------------------------
# Model
#----------------------------------------------------------------------
# Imports:
import numpy as np
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import spacy
nlp_model = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stopwords = nlp_model.Defaults.stop_words
additional_stopwords = [
    "skin", "skincare"
]
for word in additional_stopwords: stopwords.add(word)
from keybert import KeyBERT
import matplotlib.pyplot as plt
#--------------------------------------------------
# Data:
df_huda = pd.read_csv("data/whisper_huda_clean.csv")
#------------------------------
post_sent_tuple_list = []
sent_len_dict = {}

for post_idx in range(df_huda.shape[0]):
    
    post = df_huda["text_data"][post_idx].strip()
    post = re.sub(r'\b(\w+)( \1\b)+', r'\1', post) #remove repeating words in the post.
    
    sent_list = sent_tokenize(post)
    sent_len_dict[post_idx] = len(sent_list)
    post_sent_tuple_list += [(post_idx, sent, sent_idx) for sent_idx, sent in enumerate(sent_list) if len(sent.split())>5] #only add sentences with 5 or more words.
    
post_sent_tuple_df =  pd.DataFrame(post_sent_tuple_list, columns=["vid_id", "sent_text", "sent_idx"])
#------------------------------
doc_list = list(post_sent_tuple_df["sent_text"].values)
#--------------------------------------------------
# Topic model:
#------------------------------
post_sent_tuple_df = pd.read_csv("post_sent_tuple.csv")

topic_label_dict = {'topic0': 'eyeliner struggle',
 'topic1': 'different shades shades',
 'topic2': 'going ahead finish',
 'topic3': 'liquid lipsticks completely',
 'topic4': 'rose quartz palette',
 'topic5': 'energy feel vibe',
 'topic6': 'beautiful bill keece',
 'topic7': 'liner palette',
 'topic8': 'palettes going look',
 'topic9': 'reveal guys',
 'topic10': 'gloss face glossy',
 'topic11': 'glowish think pretty',
 'topic12': 'stone shade',
 'topic13': 'excited share guys',
 'topic14': 'clarifying complexion',
 'topic15': 'struggle insecurities accepting',
 'topic16': 'softens pores way',
 'topic17': 'brush applicator',
 'topic18': 'way obsessed found',
 'topic19': 'sephora',
 'topic20': 'hacks find tiktok',
 'topic21': 'try gonna try',
 'topic22': 'start contouring nose',
 'topic23': 'delete hate comments',
 'topic24': 'recycled goods',
 'topic25': 'shimmer beautiful pearls',
 'topic26': 'primer concealer eyes',
 'topic27': 'obsessed product hope',
 'topic28': 'separates gum hair',
 'topic29': 'spf inside makeup',
 'topic30': 'balm amazing ingredients',
 'topic31': 'thank guys love',
 'topic32': 'sunscreen sunblock apply',
 'topic33': 'powdered highlighter',
 'topic34': 'open grab',
 'topic35': 'beautiful turon actually',
 'topic36': 'blaming les bats',
 'topic37': 'vegetables use ugly',
 'topic38': 'draw basically arrow'}

topic_labels_0 = list(topic_label_dict.values())

topic_model = BERTopic.load("my_model")
#----------------------------------------------------------------------
# Stremlit app
#----------------------------------------------------------------------
import streamlit as st
#------------------------------
header = st.container()
viz_barchart = st.container()
viz_vids_of_a_topic = st.container()
viz_topics_in_a_vid = st.container()
#------------------------------
influ_name = "hudabeauty"
#------------------------------
with header:
    #st.title(f"Topic modeling of instagram videos of {influ_name}")
    st.title(f"HUDA BEAUTY")
#------------------------------
with viz_barchart:
    # fig_barchart = topic_model.visualize_barchart(top_n_topics=16)
    fig_barchart = topic_model.visualize_barchart(top_n_topics=8, custom_labels=True, title="Topics")
    st.plotly_chart(fig_barchart)
#------------------------------
with viz_vids_of_a_topic:

    st.title(f"Videos of a topic")

    num_tabs = 8

    tab_titles = [f"Topic {topic_idx}" for topic_idx in range(num_tabs)]
    tabs = st.tabs(tab_titles)

    for topic_idx in range(num_tabs):
        with tabs[topic_idx]:
            topic_label = topic_label_dict[f"topic{topic_idx}"]
            st.text(f"Topic {topic_idx}: {topic_label}")
            posts_5_list = list(post_sent_tuple_df[post_sent_tuple_df.topic_idx==topic_idx]["vid_id"].unique()[:5])
            
            
            # fig, axs = plt.subplots(1, 5, figsize=(30, 4))
            # for ax_idx, ax in enumerate(axs):
            #     vid_idx = posts_5_list[ax_idx]
            #     ax.text(0.5, 0.5, f"vid {vid_idx} thumbnail", fontsize=25, ha="center")
            #     ax.text(0, -0.1, df_huda.iloc[vid_idx].text_data.strip()[:20]+"...", fontsize=20)
            #     #ax.axis("off")
            #     ax.set_xticks([])
            #     ax.set_yticks([])
            # st.pyplot(fig)
            # # st.text("")

            col_list = st.columns(5)
            # vid_idx = posts_5_list[ax_idx]
            for col_idx in range(5):
                with col_list[col_idx]:
                    vid_idx = posts_5_list[col_idx]
                    huda_vid_thumbnail_name = f"thumbnails/huda_{vid_idx}_thumb.jpg"
                    st.image(image=huda_vid_thumbnail_name, width=100)
                    transcript = df_huda.iloc[vid_idx].text_data.strip()[:10]+"..."
                    st.text(transcript)
                # st.text(df_huda.iloc[vid_idx].text_data.strip()[:20]+"...", fontsize=20)


# with col1:
#    st.header("A cat")
#    st.image("https://static.streamlit.io/examples/cat.jpg")

# with col2:
#    st.header("A dog")
#    st.image("https://static.streamlit.io/examples/dog.jpg")

# with col3:
#    st.header("An owl")
#    st.image("https://static.streamlit.io/examples/owl.jpg")

#             for ax_idx, ax in enumerate(axs):
#                 vid_idx = posts_5_list[ax_idx]
#                 huda_vid_thumbnail_name = f"thumbnails/huda_{vid_idx}_thumb.jpg"
#                 st.image(image=huda_vid_thumbnail_name, width=200)

# #------------------------------
# with viz_topics_in_a_vid:

#     st.title(f"Topics in a video")

#     num_tabs = 10

#     tab_titles = [f"Video {vid_idx}" for vid_idx in range(num_tabs)]
#     tabs = st.tabs(tab_titles)

#     for vid_idx in range(num_tabs):
#         with tabs[vid_idx]:
#             this_vid_df = post_sent_tuple_df[post_sent_tuple_df.vid_id==vid_idx]
            
#             topic_idx_list = list(this_vid_df.topic_idx.unique())
#             if -1 in topic_idx_list: topic_idx_list.remove(-1)
#             topic_idx_list.sort()

#             st.subheader(f"Video {vid_idx}")

#             # #-----
#             # empty_fig, empty_axs = plt.subplots(1, sent_len_dict[vid_idx], figsize=(10, 0.5), squeeze=True)
#             # for ax in empty_axs:
#             #     ax.set_xticks([])
#             #     ax.set_yticks([])
#             # #-----

#             if len(topic_idx_list)==0:
#                 st.text("No topics found!")

#             else:
#                 nested_tab_titles = [f"Topic {topic_idx}" for topic_idx in range(len(topic_idx_list))]
#                 nested_tabs = st.tabs(nested_tab_titles)

#                 #fig_list = []
#                 #axs_list = []
#                 topic_label_list = []
#                 for tt, topic_idx in enumerate(topic_idx_list):
#                     topic_label_list.append(f"{topic_label_dict[f'topic{topic_idx}']}")
#                     fig, axs = plt.subplots(1, sent_len_dict[vid_idx], figsize=(10, 0.5), squeeze=True)

#                     ax_idx_list = list(this_vid_df[this_vid_df.topic_idx==topic_idx].sent_idx.values)
#                     this_sent_list = this_vid_df[this_vid_df.sent_idx.isin(ax_idx_list)].sent_text.values
                    
#                     #if len(axs)>1:
#                     if type(axs)==type(np.array([1])) and len(axs)>1:
#                         for ax_idx in ax_idx_list:
#                             axs[ax_idx].set_facecolor("black")

#                         for ax in axs:
#                             ax.set_xticks([])
#                             ax.set_yticks([])
#                     else:
#                         axs.set_facecolor("black")

#                         axs.set_xticks([])
#                         axs.set_yticks([])
                    
#                     #fig_list.append(fig)
#                     #axs_list.append(axs)
#                     plt.close()
#                     with nested_tabs[tt]:
#                         topic_label = topic_label_list[tt]
#                         st.text(topic_label)
#                         st.pyplot(fig)
#                         for sent in this_sent_list:
#                             st.write(sent)
#------------------------------
with viz_topics_in_a_vid:

    st.title(f"Topics in a video")

    num_tabs = 10

    tab_titles = [f"Video {vid_idx}" for vid_idx in range(num_tabs)]
    tabs = st.tabs(tab_titles)

    for vid_idx in range(num_tabs):
        with tabs[vid_idx]:
            this_vid_df = post_sent_tuple_df[post_sent_tuple_df.vid_id==vid_idx]
            
            topic_idx_list = list(this_vid_df.topic_idx.unique())
            if -1 in topic_idx_list: topic_idx_list.remove(-1)
            topic_idx_list.sort()

            st.subheader(f"Video {vid_idx}")
            #st.text("put thumbnail here")
            huda_vid_thumbnail_name = f"thumbnails/huda_{vid_idx}_thumb.jpg"
            st.image(image=huda_vid_thumbnail_name, width=200)

            # #-----
            # empty_fig, empty_axs = plt.subplots(1, sent_len_dict[vid_idx], figsize=(10, 0.5), squeeze=True)
            # for ax in empty_axs:
            #     ax.set_xticks([])
            #     ax.set_yticks([])
            # #-----

            if len(topic_idx_list)==0:
                st.text("No topics found!")

            else:
                nested_tab_titles = [f"Topic {topic_idx}" for topic_idx in range(len(topic_idx_list))]

                # for tt, topic in enumerate(nested_tab_titles):
                    
                #     ax_idx_list = list(this_vid_df[this_vid_df.topic_idx==topic_idx].sent_idx.values)
                # this_sent_list = this_vid_df[this_vid_df.sent_idx.isin(ax_idx_list)].sent_text.values

                # st.text(f"Topic: {topic_labels_0[tt]}")

                for topic_idx in topic_idx_list:
                    
                    ax_idx_list = list(this_vid_df[this_vid_df.topic_idx==topic_idx].sent_idx.values)
                    this_sent_list = this_vid_df[this_vid_df.sent_idx.isin(ax_idx_list)].sent_text.values

                    st.subheader(f"Topic {topic_idx}: {topic_labels_0[topic_idx]}")
                    for sent in this_sent_list:
                        st.markdown(sent)



                # nested_tabs = st.tabs(nested_tab_titles)

                # #fig_list = []
                # #axs_list = []
                # topic_label_list = []
                # for tt, topic_idx in enumerate(topic_idx_list):
                #     topic_label_list.append(f"{topic_label_dict[f'topic{topic_idx}']}")
                #     fig, axs = plt.subplots(1, sent_len_dict[vid_idx], figsize=(10, 0.5), squeeze=True)

                #     ax_idx_list = list(this_vid_df[this_vid_df.topic_idx==topic_idx].sent_idx.values)
                #     this_sent_list = this_vid_df[this_vid_df.sent_idx.isin(ax_idx_list)].sent_text.values
                    
                #     #if len(axs)>1:
                #     if type(axs)==type(np.array([1])) and len(axs)>1:
                #         for ax_idx in ax_idx_list:
                #             axs[ax_idx].set_facecolor("black")

                #         for ax in axs:
                #             ax.set_xticks([])
                #             ax.set_yticks([])
                #     else:
                #         axs.set_facecolor("black")

                #         axs.set_xticks([])
                #         axs.set_yticks([])
                    
                #     #fig_list.append(fig)
                #     #axs_list.append(axs)
                #     plt.close()
                #     with nested_tabs[tt]:
                #         topic_label = topic_label_list[tt]
                #         st.text(topic_label)
                #         st.pyplot(fig)
                #         for sent in this_sent_list:
                #             st.write(sent)