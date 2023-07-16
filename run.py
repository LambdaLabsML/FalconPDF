import os
from datetime import datetime
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

# model_names = ["tiiuae/falcon-7b-instruct", "tiiuae/falcon-40b-instruct", "tiiuae/falcon-rw-1b"]
model_names = ["tiiuae/falcon-7b-instruct"]
embedding_function_name = "all-mpnet-base-v2"
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
max_new_tokens = 1024
repetition_penalty = 10.0
temperature = 0
stop_words_ids = [[193, 7932, 37]]
chunk_size = 512
chunk_overlap = 32

class StopOnWords(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

def get_uuid():
    return str(uuid4())


def create_embedding_function(embedding_function_name):
    return HuggingFaceEmbeddings(model_name=embedding_function_name,
                                 model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})

def create_pipelines(model_names):
    pipelines = {}
    
    for model_name in model_names:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        stop_words_ids_pt = [torch.tensor(x) for x in stop_words_ids]
        stopping_criteria = StoppingCriteriaList([StopOnWords(stops=stop_words_ids_pt)])

        if model_name == "tiiuae/falcon-40b-instruct":
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
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )

        model.eval()
        print(f"Model loaded on {device}")

        generate_text = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            stopping_criteria=stopping_criteria,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty
        )

        pipelines[model_name] = HuggingFacePipeline(pipeline=generate_text)

    return pipelines

pipelines = create_pipelines(model_names)
embedding_function = create_embedding_function(embedding_function_name)

def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, None]]


def bot(model_name, db_path, chat_mode, history):
    if not history or history[-1][0] == "":
        gr.Info("Please start the conversation by saying something.")
        return None

    chat_hist = history[:-1]
    if chat_hist:
        chat_hist = [tuple([y.replace("\n", ' ').strip(" ") for y in x]) for x in chat_hist]

    print("@" * 20)
    print(f"chat_hist:\n {chat_hist}")
    print("@" * 20)

    print('------------------------------------')
    print(model_name)
    print(db_path)
    print(chat_mode)
    print('------------------------------------')

    # Need to create langchain model from db for each session
    db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
    if chat_mode.lower() == 'basic':
        print("chat mode: basic")
        qa = RetrievalQA.from_llm(
            llm=pipelines[model_name],
            retriever=db.as_retriever(),
            return_source_documents=True
        )
        response = qa(
            {"query": history[-1][0]}
        )
        history[-1][1] = response['result']
    else:
        print("chat mode: conversational")
        qa = ConversationalRetrievalChain.from_llm(
            llm=pipelines[model_name],
            retriever=db.as_retriever(),
            return_source_documents=True
        )
        response = qa(
            {"question": history[-1][0],
             "chat_history": chat_hist,
             })
        history[-1][1] = response['answer']

    print(response)

    return history


def pdf_changes(pdf_doc):
    print("pdf changes, loading documents")

    # Persistently store the db next to the uploaded pdf
    db_path, file_ext = os.path.splitext(pdf_doc.name)

    timestamp = datetime.now()
    db_path += "_" + timestamp.strftime("%Y-%m-%d-%H-%S")

    loader = OnlinePDFLoader(pdf_doc.name)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    
    db = Chroma.from_documents(texts, embedding_function, persist_directory=db_path)
    db.persist()
    return db_path


def init():
    with gr.Blocks(
            theme=gr.themes.Soft(),
            css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        gr.HTML(
            """
                <div style="text-align: center; max-width: 650px; margin: 0 auto;">
                  <div>
                    <img class="logo" src="https://lambdalabs.com/hubfs/logos/lambda-logo.svg" alt="Lambda Logo"
                        style="margin: auto; max-width: 7rem;">
                    <h1 style="font-weight: 900; font-size: 3rem;">
                      Chat With FalconPDF
                    </h1>
                  </div>
                </div>
            """
        )

        pdf_doc = gr.File(label="Load a pdf", file_types=['.pdf'], type="file")
        model_id = gr.Radio(label="LLM", choices=model_names, value=model_names[0], interactive=True)
        db_path = gr.Textbox(label="DB_PATH", visible=False)
        chat_mode = gr.Radio(label="Chat mode", choices=['Basic', 'Conversational'], value='Basic',
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

        model_id.change(clear_input, inputs=[], outputs=[msg])

        pdf_doc.upload(pdf_changes, inputs=[pdf_doc], outputs=[db_path]). \
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
                model_id,
                db_path,
                chat_mode,
                chatbot,
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
                model_id,
                db_path,
                chat_mode,
                chatbot,
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

    demo.queue(max_size=4, concurrency_count=2)

    demo.launch(server_port=8266, inline=False, share=True)


if __name__ == "__main__":
    init()
