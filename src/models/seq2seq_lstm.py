"""
Seq2Seq LSTM model for deterministic sequence prediction.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    LSTM Encoder that reads input sequences and produces a context vector.

    Args:
        input_size: Number of input features
        hidden_size: Size of hidden state
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            outputs: LSTM outputs (batch_size, seq_len, hidden_size)
            hidden: Tuple of (h_n, c_n) hidden states
        """
        outputs, hidden = self.lstm(x)
        return outputs, hidden


class Decoder(nn.Module):
    """
    LSTM Decoder that generates output sequences from context.

    Args:
        output_size: Number of output features (1 for clicks)
        hidden_size: Size of hidden state
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        output_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, output_size)
            hidden: Tuple of (h, c) hidden states

        Returns:
            output: Predicted value (batch_size, 1, output_size)
            hidden: Updated hidden state
        """
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden


class Seq2SeqLSTM(nn.Module):
    """
    Complete Seq2Seq LSTM model for deterministic prediction.

    Architecture:
    - Encoder: Reads past 4 weeks of features
    - Decoder: Generates future 2 weeks of clicks (single path)

    Args:
        input_size: Number of input features
        hidden_size: Size of hidden state
        output_size: Number of output features (1 for clicks)
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(Seq2SeqLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(output_size, hidden_size, num_layers, dropout)

    def forward(self, src, tgt_len, teacher_forcing_ratio=0.5):
        """
        Args:
            src: Source sequence (batch_size, src_len, input_size)
            tgt_len: Target sequence length (number of future weeks)
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            outputs: Predicted sequence (batch_size, tgt_len, output_size)
        """
        batch_size = src.size(0)

        # Encode input sequence
        _, hidden = self.encoder(src)

        # Initialize decoder input (zeros for first prediction)
        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=src.device)

        # Store predictions
        outputs = []

        # Generate output sequence step by step
        for t in range(tgt_len):
            # Decode one step
            output, hidden = self.decoder(decoder_input, hidden)
            outputs.append(output)

            # Next input is current prediction (no teacher forcing in inference)
            decoder_input = output

        # Concatenate all predictions
        outputs = torch.cat(outputs, dim=1)  # (batch_size, tgt_len, output_size)

        return outputs

    def predict(self, src, tgt_len):
        """
        Inference mode - single deterministic prediction.

        Args:
            src: Source sequence (batch_size, src_len, input_size)
            tgt_len: Target sequence length

        Returns:
            Predicted sequence (batch_size, tgt_len, output_size)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(src, tgt_len, teacher_forcing_ratio=0.0)
