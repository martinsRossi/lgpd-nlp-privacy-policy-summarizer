"""
Módulo de Sumarização Abstrativa
Implementa sumarização abstrativa usando modelos Transformer (T5, GPT-2)
"""

from typing import Dict, Optional, List
from loguru import logger
import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    pipeline
)


class SumarizadorAbstrativo:
    """Classe para sumarização abstrativa usando modelos de linguagem"""
    
    def __init__(
        self,
        modelo: str = 't5-small',
        dispositivo: Optional[str] = None
    ):
        """
        Inicializa o sumarizador abstrativo
        
        Args:
            modelo: Nome do modelo ('t5-small', 't5-base', 'gpt2', etc)
            dispositivo: Dispositivo ('cpu', 'cuda', None=auto)
        """
        self.modelo_nome = modelo
        
        # Detectar dispositivo
        if dispositivo is None:
            self.dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.dispositivo = dispositivo
        
        logger.info(f"Inicializando SumarizadorAbstrativo (modelo: {modelo}, dispositivo: {self.dispositivo})")
        
        self.tokenizer = None
        self.modelo = None
        self.pipeline = None
        
        self._carregar_modelo()
    
    def _carregar_modelo(self):
        """Carrega o modelo e tokenizer"""
        try:
            if 't5' in self.modelo_nome.lower() or 'ptt5' in self.modelo_nome.lower():
                # Modelos T5 (português e inglês)
                if 'portuguese' in self.modelo_nome or 'ptt5' in self.modelo_nome:
                    # PTT5 - T5 otimizado para português
                    logger.info("Carregando PTT5 (otimizado para português)...")
                    self.tokenizer = AutoTokenizer.from_pretrained('unicamp-dl/ptt5-small-portuguese-vocab')
                    self.modelo = AutoModelForSeq2SeqLM.from_pretrained('unicamp-dl/ptt5-small-portuguese-vocab')
                    logger.success("PTT5 carregado com sucesso!")
                else:
                    # T5 padrão (inglês)
                    logger.info("Carregando T5-small (inglês)...")
                    self.tokenizer = T5Tokenizer.from_pretrained(self.modelo_nome)
                    self.modelo = T5ForConditionalGeneration.from_pretrained(self.modelo_nome)
                    logger.warning("T5-small não é otimizado para português - qualidade limitada")
                
                self.modelo.to(self.dispositivo)
                logger.success(f"Modelo T5 carregado: {self.modelo_nome}")
            
            elif 'gpt' in self.modelo_nome.lower():
                # Modelos GPT-2
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.modelo_nome)
                self.modelo = GPT2LMHeadModel.from_pretrained(self.modelo_nome)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.modelo.to(self.dispositivo)
                logger.success(f"Modelo GPT-2 carregado: {self.modelo_nome}")
            
            else:
                # Tentar pipeline genérico
                self.pipeline = pipeline(
                    'summarization',
                    model=self.modelo_nome,
                    device=0 if self.dispositivo == 'cuda' else -1
                )
                logger.success(f"Pipeline de sumarização carregado: {self.modelo_nome}")
        
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {self.modelo_nome}: {e}")
            logger.warning("Usando modo fallback sem modelo")
            self.modelo = None
    
    def sumarizar_t5(
        self,
        texto: str,
        max_length: int = 150,
        min_length: int = 40,
        num_beams: int = 4
    ) -> str:
        """
        Sumarização usando T5
        
        Args:
            texto: Texto a sumarizar
            max_length: Comprimento máximo do sumário
            min_length: Comprimento mínimo do sumário
            num_beams: Número de beams para beam search
            
        Returns:
            Texto sumarizado
        """
        if self.modelo is None or self.tokenizer is None:
            logger.error("Modelo T5 não carregado")
            return texto[:max_length]
        
        # Preparar entrada
        if 'ptt5' in self.modelo_nome.lower() or 'portuguese' in self.modelo_nome.lower():
            # PTT5 usa "sumarize" (sem o segundo 'm')
            input_text = f"sumarize: {texto}"
            logger.info("Usando prompt PTT5: 'sumarize:'")
        else:
            # T5 padrão usa "summarize"
            input_text = f"summarize: {texto}"
            logger.info("Usando prompt T5: 'summarize:'")
        
        # Tokenizar
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.dispositivo)
        
        # Gerar sumário
        with torch.no_grad():
            summary_ids = self.modelo.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,  # Reduzido para evitar forçar tokens inválidos
                repetition_penalty=1.2,  # Penaliza repetições moderadamente (era 2.0)
                length_penalty=1.0,      # Mantém comprimento balanceado
                do_sample=False,         # Gera deterministicamente (beam search)
                forced_eos_token_id=self.tokenizer.eos_token_id  # Força parada no EOS
            )
        
        # Decodificar
        sumario = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Remover prompt se aparecer no output (bug do PTT5)
        if sumario.startswith('sumarize:'):
            sumario = sumario.replace('sumarize:', '', 1).strip()
        elif sumario.startswith('summarize:'):
            sumario = sumario.replace('summarize:', '', 1).strip()
        
        # Pós-processamento: limpar lixo/corrupção no final
        sumario = self._limpar_sumario(sumario)
        
        logger.info(f"Sumário T5 gerado: {len(sumario)} caracteres")
        return sumario
    
    def _limpar_sumario(self, texto: str) -> str:
        """
        Remove lixo/corrupção do sumário gerado
        
        Args:
            texto: Texto a limpar
            
        Returns:
            Texto limpo
        """
        import re
        
        # 0. Remover números de seção no início (ex: "1.1 ", "2.3 ")
        texto = re.sub(r'^\d+\.\d+\s+', '', texto)
        texto = re.sub(r'^\d+\.\s+', '', texto)
        
        # 1. Detectar onde começa a corrupção (caracteres especiais estranhos)
        # Procura por padrões de corrupção como: »â€" ou múltiplos símbolos
        match_corrupcao = re.search(r'[»€"—]{2,}|[^\w\s\.,;:!?()\[\]"\'À-ÿ-]{3,}', texto)
        if match_corrupcao:
            # Truncar antes da corrupção
            texto = texto[:match_corrupcao.start()].strip()
        
        # 2. Remover sequências repetidas de palavras curtas (ex: "e e e e")
        texto = re.sub(r'\b(\w{1,2})\s+\1\s+\1', r'\1', texto)
        
        # 3. Remover fim malformado (palavras muito curtas + símbolos)
        texto = re.sub(r'\s+[a-z]{1,3}(\s+[^\w\s]+)+\s*$', '', texto)
        
        # 4. Remover caracteres especiais no final
        texto = re.sub(r'[»â€"?:;&\']+\s*$', '', texto)
        
        # 5. Limpar pontuação múltipla
        texto = re.sub(r'\.{2,}', '.', texto)
        texto = re.sub(r',{2,}', ',', texto)
        
        # 6. Se termina com "and" ou palavra incompleta, remover
        texto = re.sub(r'\s+(and|ou|e|de|da|do)\s*$', '', texto, flags=re.IGNORECASE)
        
        # 7. Garantir que termina com pontuação
        if texto and not texto[-1] in '.!?':
            palavras = texto.split()
            # Só adiciona ponto se a última palavra parece válida
            if palavras and len(palavras[-1]) >= 3 and palavras[-1].isalnum():
                texto += '.'
            else:
                # Remove última palavra se parece corrompida
                texto = ' '.join(palavras[:-1]) + '.'
        
        return texto.strip()
        return sumario
    
    def sumarizar_gpt2(
        self,
        texto: str,
        max_length: int = 150,
        temperature: float = 0.7
    ) -> str:
        """
        Sumarização usando GPT-2 (mais experimental)
        
        Args:
            texto: Texto a sumarizar
            max_length: Comprimento máximo
            temperature: Temperatura de geração
            
        Returns:
            Texto sumarizado
        """
        if self.modelo is None or self.tokenizer is None:
            logger.error("Modelo GPT-2 não carregado")
            return texto[:max_length]
        
        # Criar prompt
        prompt = f"Resuma o seguinte texto:\n\n{texto}\n\nResumo:"
        
        # Tokenizar
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.dispositivo)
        
        # Gerar
        with torch.no_grad():
            outputs = self.modelo.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # Evita repetição de 3-gramas
                repetition_penalty=1.5,  # Penaliza repetições
                top_k=50,                # Top-K sampling
                top_p=0.95              # Nucleus sampling
            )
        
        # Decodificar e extrair apenas o sumário
        texto_completo = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Tentar extrair apenas a parte após "Resumo:"
        if "Resumo:" in texto_completo:
            sumario = texto_completo.split("Resumo:")[-1].strip()
        else:
            sumario = texto_completo[len(prompt):].strip()
        
        logger.info(f"Sumário GPT-2 gerado: {len(sumario)} caracteres")
        return sumario
    
    def sumarizar_pipeline(
        self,
        texto: str,
        max_length: int = 150,
        min_length: int = 40
    ) -> str:
        """
        Sumarização usando pipeline do HuggingFace
        
        Args:
            texto: Texto a sumarizar
            max_length: Comprimento máximo
            min_length: Comprimento mínimo
            
        Returns:
            Texto sumarizado
        """
        if self.pipeline is None:
            logger.error("Pipeline não disponível")
            return texto[:max_length]
        
        resultado = self.pipeline(
            texto,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        
        sumario = resultado[0]['summary_text']
        logger.info(f"Sumário (pipeline) gerado: {len(sumario)} caracteres")
        return sumario
    
    def sumarizar(
        self,
        texto: str,
        max_length: int = 150,
        min_length: int = 40,
        num_beams: int = 4
    ) -> Dict:
        """
        Método principal de sumarização abstrativa
        
        Args:
            texto: Texto completo
            max_length: Comprimento máximo do sumário
            min_length: Comprimento mínimo do sumário
            num_beams: Número de beams (para T5)
            
        Returns:
            Dicionário com sumário e metadados
        """
        logger.info(f"Iniciando sumarização abstrativa (modelo: {self.modelo_nome})")
        
        # Escolher método baseado no modelo
        try:
            if 't5' in self.modelo_nome.lower():
                sumario = self.sumarizar_t5(texto, max_length, min_length, num_beams)
            elif 'gpt' in self.modelo_nome.lower():
                sumario = self.sumarizar_gpt2(texto, max_length)
            elif self.pipeline is not None:
                sumario = self.sumarizar_pipeline(texto, max_length, min_length)
            else:
                logger.error("Nenhum método de sumarização disponível")
                # Fallback: truncar texto
                sumario = texto[:max_length] + "..."
        except Exception as e:
            logger.error(f"Erro na sumarização: {e}")
            sumario = texto[:max_length] + "..."
        
        # Calcular métricas
        taxa_compressao = len(sumario) / len(texto) if len(texto) > 0 else 0
        
        resultado = {
            'sumario': sumario,
            'texto_original': texto,
            'modelo': self.modelo_nome,
            'num_caracteres_original': len(texto),
            'num_caracteres_sumario': len(sumario),
            'taxa_compressao_caracteres': taxa_compressao
        }
        
        logger.success(f"Sumarização abstrativa concluída ({taxa_compressao:.1%} de compressão)")
        return resultado
    
    def sumarizar_lote(
        self,
        textos: List[str],
        max_length: int = 150,
        min_length: int = 40
    ) -> List[Dict]:
        """
        Sumariza múltiplos textos
        
        Args:
            textos: Lista de textos
            max_length: Comprimento máximo
            min_length: Comprimento mínimo
            
        Returns:
            Lista de dicionários com sumários
        """
        return [
            self.sumarizar(texto, max_length, min_length)
            for texto in textos
        ]
    
    def liberar_memoria(self):
        """Libera memória GPU"""
        if self.modelo is not None:
            del self.modelo
            del self.tokenizer
        if self.pipeline is not None:
            del self.pipeline
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Memória liberada")
