import { Tiktoken } from 'js-tiktoken/lite';
import o200k from 'js-tiktoken/ranks/o200k_base';
window.DeckTokenizer = new Tiktoken(o200k);
