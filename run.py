from uuid import uuid4
import gradio as gr
from time import sleep
import torch
from torch import cuda, bfloat16
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline

model_names = ["tiiuae/falcon-7b-instruct", "tiiuae/falcon-40b-instruct"]
model_name = None
model = None
pipeline = None
tokenizer = None
stop_token_ids = []
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

db = None
qa = None

USER_NAME = "Human"
BOT_NAME = "AI"
max_new_tokens = 1024

STOP_STR = f"\n{USER_NAME}:"
STOP_SUSPECT_LIST = [":", "\n", "User"]


def load_falcon(selected_model_name, progress=gr.Progress(track_tqdm=True)):
    global model, model_name, stop_token_ids, pipeline

    if model is None or model_name != selected_model_name:

        progress(0, desc="Loading model ...")

        if selected_model_name == model_names[1]:
            model_name = model_names[1]
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bfloat16
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map='auto'
            )
        else:
            model_name = model_names[0]
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )

        model.eval()
        print(f"Model loaded on {device}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        stop_token_ids = [
            tokenizer.convert_tokens_to_ids(x) for x in [
                [USER_NAME, ':'], [BOT_NAME, ':']
            ]
        ]

        stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

        generate_text = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            stopping_criteria=stopping_criteria,  # without this model rambles during chat
            temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # mex number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
        )

        pipeline = HuggingFacePipeline(pipeline=generate_text)

    return "Model ready!", gr.update(interactive=True)


def check_stopwords(list_key, input_str):
    input_str_lower = input_str.lower()
    matched_indices = []
    for keyword in list_key:
        keyword_lower = keyword.lower()
        start = 0
        while start < len(input_str_lower):
            index = input_str_lower.find(keyword_lower, start)
            if index == -1:
                break
            end = index + len(keyword_lower) - 1
            matched_indices.append((index, end))
            start = end + 1
    return len(matched_indices) > 0, matched_indices


class StopOnWords(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False


stop_words = ["\nUser:", ]
stop_words_ids = [[193, 7932, 37]]

stop_words_ids_pt = [torch.tensor(x) for x in stop_words_ids]

stopping_criteria = StoppingCriteriaList([StopOnWords(stops=stop_words_ids_pt)])


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, None]]


def bot(history, retriever):
    if not history or history[-1][0] == "":
        gr.Info("Please start the conversation by saying something.")
        return None

    global qa, pipeline
    if pipeline is None:
        gr.Info("Please load a model first.")
        return None

    if qa is None:
        gr.Info("Please upload a pdf file first.")
        return None

    chat_hist = history[:-1]
    if chat_hist:
        chat_hist = [tuple([y.replace("\n", ' ').strip(" ") for y in x]) for x in chat_hist]

    print("@" * 20)
    print(f"chat_hist:\n {chat_hist}")
    print("@" * 20)

    if retriever.lower() == 'basic':
        response = qa(
            {"query": history[-1][0]})
    else:
        response = qa(
            {"question": history[-1][0],
             "chat_history": chat_hist,
             })

    history[-1][1] = response['answer' if retriever.lower() == 'conversational' else 'result']

    return history


def get_uuid():
    return str(uuid4())


def create_sbert_mpnet():
    EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(model_name=EMB_SBERT_MPNET_BASE, model_kwargs={"device": device})


def retriever_changes(r):
    global db, qa, pipeline
    if db is None:
        return

    if r.lower() == 'basic':
        if not isinstance(qa, RetrievalQA):
            print("retriever changes: basic")
            qa = RetrievalQA.from_llm(llm=pipeline, retriever=db.as_retriever(), return_source_documents=True)
    else:
        if not isinstance(qa, ConversationalRetrievalChain):
            print("retriever changes: conversational")
            qa = ConversationalRetrievalChain.from_llm(llm=pipeline, retriever=db.as_retriever(),
                                                       return_source_documents=True)


def pdf_changes(pdf_doc, r):
    print("pdf changes, loading documents")
    loader = OnlinePDFLoader(pdf_doc.name)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = create_sbert_mpnet()
    global db
    db = Chroma.from_documents(texts, embeddings)
    retriever_changes(r)
    return None


def init():
    with gr.Blocks(
            theme=gr.themes.Soft(),
            css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        conversation_id = gr.State(get_uuid)

        gr.HTML(
            """
                <div style="text-align: center; max-width: 650px; margin: 0 auto;">
                  <div>
                    <img class="logo" src="https://lambdalabs.com/hubfs/logos/lambda-logo.svg" alt="Lambda Logo"
                        style="margin: auto; max-width: 7rem;">
                    <h1 style="font-weight: 900; font-size: 3rem;">
                      Chat With Falcon
                    </h1>
                  </div>
                </div>
            """
        )

        pdf_doc = gr.File(label="Load a pdf", file_types=['.pdf'], type="file")
        model_id = gr.Radio(label="LLM", choices=model_names, value=model_names[0])
        retriever = gr.Radio(label="Chat mode", choices=['Basic', 'Conversational'], value='Basic',
                             info="Basic: no coversational context. Conversational: uses conversational context.")
        chatbot = gr.Chatbot(height=500)

        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Chat Message Box",
                    placeholder="Chat Message Box",
                    show_label=False,
                    container=False
                )
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")

        gr.Examples(['What is the summary of the document?',
                     'What is the motivation of the document?'],
                    inputs=msg)

        def clear_input():
            sleep(1)
            return ""

        with gr.Row():
            gr.HTML(
                """
                    <div class="footer">
                        <p> A chatbot tries to give helpful, detailed, and polite answers to the user's questions. Gradio Demo created by <a href="https://lambdalabs.com/">Lambda</a>.</p>
                    </div>
                    <div class="acknowledgments">
                        <p> It is based on Falcon 7B/40B. More information can be found <a href="https://falconllm.tii.ae/">here</a>.</p>
                    </div>
                """
            )

        def frozen():
            return gr.update(interactive=False)

        model_id.change(frozen, inputs=[], outputs=[model_id]) \
            .then(load_falcon, inputs=[model_id], outputs=[msg, model_id]) \
            .then(clear_input, inputs=[], outputs=[msg])

        retriever.change(retriever_changes, inputs=[retriever], outputs=[])

        pdf_doc.upload(pdf_changes, inputs=[pdf_doc, retriever], outputs=[msg]). \
            then(clear_input, inputs=[], outputs=[msg]). \
            then(lambda: None, None, chatbot)

        # enter key event
        submit_event = msg.submit(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[
                chatbot,
                retriever
            ],
            outputs=chatbot,
            queue=True,
        )

        # click submit button event
        submit_click_event = submit.click(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[
                chatbot,
                retriever
            ],
            outputs=chatbot,
            queue=True,
        )

        stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[submit_event, submit_click_event],
            queue=False,
        )

        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue(max_size=128, concurrency_count=2)

    demo.launch(server_port=8266, inline=False, share=True)


if __name__ == "__main__":
    load_falcon(model_names[0])
    init()
